"""Summarize fixed C interventions; never fit or select a deployed model."""
import csv
import json
from pathlib import Path
import numpy as np
from relation_experiment import ROOT, dump, sha, timestamp
from reference_selection_experiment import read
from reference_selection_space import analyze_arrays


def main():
    out = ROOT/'Result/c_disagreement_diagnosis'
    assert read(out/'status.json')['state'] == 'complete'
    assert read(out/'audit.json')['passed']
    plan, records = read(out/'PLAN.json'), read(out/'records.json')
    assert sha(ROOT/'diagnose_c_disagreements.py') == plan['diagnostic_source_sha256']
    for path, digest in read(ROOT/'Result/reference_selection/PROTOCOL.json')['code_sha256'].items():
        assert sha(ROOT/path) == digest
    arrays, seeds, folds = [], [], []
    for entry, record in zip(plan['entries'], records):
        assert entry['seed'] == record['seed'] and entry['fold'] == record['fold']
        for key in ('checkpoint', 'c_predictions', 'self_predictions'):
            assert sha(Path(entry[key])) == entry[key+'_sha256']
        folder = out/f"seed_{entry['seed']}"/f"fold_{entry['fold']:02d}"
        assert sha(folder/'predictions.npz') == record['predictions_sha256']
        with np.load(folder/'predictions.npz') as saved:
            arrays.append({key: saved[key].copy() for key in saved.files})
        seeds.extend([entry['seed']]*len(arrays[-1]['labels']))
        folds.extend([entry['fold']]*len(arrays[-1]['labels']))
    names = arrays[0]['condition_names'].tolist()
    for a in arrays:
        assert a['condition_names'].tolist() == names
        np.testing.assert_array_equal(a['masks'], arrays[0]['masks'])
    logits = np.concatenate([a['condition_logits'] for a in arrays], axis=1)
    mask_logits = np.concatenate([a['mask_logits'] for a in arrays], axis=1)
    losses = np.concatenate([a['mask_losses'] for a in arrays], axis=1)
    labels = np.concatenate([a['labels'] for a in arrays])
    groups = np.concatenate([a['group'] for a in arrays])
    rows = np.concatenate([a['source_row'] for a in arrays])
    self_pred = np.concatenate([a['self_predictions'] for a in arrays])
    seeds, folds = np.asarray(seeds), np.asarray(folds)
    pred = logits.argmax(-1)
    correct = pred == labels[None]
    baseline = correct[names.index('all_open')]
    mask_correct = mask_logits.argmax(-1) == labels[None]
    any_binary = mask_correct.any(0)
    total = len(labels)
    assert total == 1794 and len(np.unique(rows)) == 598

    def summarize(selected):
        n = int(selected.sum())
        bc = baseline[selected]
        output = dict(evaluations=n, distinct_patient_rows=len(np.unique(rows[selected])),
            base_correct=int(bc.sum()), base_errors=int((~bc).sum()), conditions={})
        for k, name in enumerate(names):
            cc = correct[k, selected]
            output['conditions'][name] = dict(correct=int(cc.sum()),
                recovered=int((~bc & cc).sum()), harmed=int((bc & ~cc).sum()),
                changed=int((pred[k, selected] != pred[0, selected]).sum()))
        content_names = [name for name in names if name.endswith('_norm_matched')]
        content_any = correct[[names.index(name) for name in content_names]].any(0)
        permutation_names = [f'permutation_{k}_norm_matched' for k in range(3)]
        permutation_all = correct[[names.index(name) for name in permutation_names]].all(0)
        all_closed = correct[names.index('all_closed')]
        bypass = correct[names.index('bypass_graph_update')]
        any_tested = any_binary | correct.any(0)
        wrong = selected & ~baseline
        output['diagnostic_recovery_sets'] = dict(
            any_binary=int((wrong & any_binary).sum()),
            binary_but_not_all_closed=int((wrong & any_binary & ~all_closed).sum()),
            bypass_but_not_any_binary=int((wrong & bypass & ~any_binary).sum()),
            norm_matched_content_any=int((wrong & content_any).sum()),
            norm_matched_permutation_all_three=int((wrong & permutation_all).sum()),
            content_but_not_any_binary=int((wrong & content_any & ~any_binary).sum()),
            any_tested=int((wrong & any_tested).sum()),
            none_of_tested=int((wrong & ~any_tested).sum()))
        return output

    summaries = {name: summarize(groups == name) for name in ('c_wrong_self_right', 'both_wrong', 'c_right_self_wrong', 'both_right')}
    expected = read(ROOT/'Result/reference_selection_replication/已有C与无图错误核对.json')['total']
    assert summaries['c_wrong_self_right']['evaluations'] == expected['c_wrong_self_right'] == 41
    assert summaries['both_wrong']['evaluations'] == expected['both_wrong'] == 64
    all_summary = summarize(np.ones(total, dtype=bool))
    assert all_summary['conditions']['all_closed']['recovered'] == 9
    assert all_summary['conditions']['all_closed']['harmed'] == 96
    oracle, _ = analyze_arrays(mask_logits, losses, labels, arrays[0]['masks'])
    by_seed = {}
    for seed in (0, 1, 2):
        mask = seeds == seed
        seed_oracle, _ = analyze_arrays(mask_logits[:, mask], losses[:, mask], labels[mask], arrays[0]['masks'])
        by_seed[str(seed)] = dict(all=summarize(mask), target=summarize(mask & (groups == 'c_wrong_self_right')),
                                 mask_oracle=seed_oracle)
    result = dict(analyzed_at=timestamp(), all=all_summary, groups=summaries, by_seed=by_seed,
        by_class={name: summarize(labels == i) for i, name in enumerate(('AD', 'CN', 'SMCI'))},
        mask_oracle=oracle, source_sha256=sha(Path(__file__)),
        caveats=plan['caveats'], mean_fold_acc_by_condition={
            name:float(np.mean([correct[k, (seeds==s)&(folds==f)].mean() for s in (0,1,2) for f in range(1,11)]))
            for k,name in enumerate(names)})
    dump(out/'analysis.json', result)
    patient_records = []
    for j in range(total):
        r = dict(seed=int(seeds[j]), fold=int(folds[j]), source_row=int(rows[j]),
            true_class=int(labels[j]), group=str(groups[j]), c_prediction=int(pred[0,j]),
            self_prediction=int(self_pred[j]), any_binary_correct=int(any_binary[j]),
            number_of_correct_binary_masks=int(mask_correct[:,j].sum()),
            best_possible_nll=float(losses[:,j].min()), all_open_nll=float(losses[-1,j]))
        r.update({name+'_prediction':int(pred[k,j]) for k,name in enumerate(names)})
        patient_records.append(r)
    for filename, selected in [('逐患者干预.csv', patient_records),
        ('无图对而C错的病例.csv', [r for r in patient_records if r['group']=='c_wrong_self_right'])]:
        with (out/filename).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(patient_records[0]))
            writer.writeheader()
            writer.writerows(selected)
    # Reports use counts instead of presenting an oracle as a model improvement.
    target = summaries['c_wrong_self_right']
    tc = target['conditions']
    tr = target['diagnostic_recovery_sets']
    text = ['# C 与无图模型分歧的固定模型干预', '',
        '**结论：本轮没有支持“无图对、C 错主要是邻居内容干扰造成”的解释。**'
        f"目标组 {target['evaluations']} 次错误中，模态参照开关最多只能纠正 {tr['any_binary']} 次，"
        f"这些都可由全部关闭参照纠正；{tr['none_of_tested']} 次在本轮所有干预下仍然错误。"
        '未被纠正并不证明问题一定出在自身表征，只表示当前这些推断期操作不能解释或修复大多数分歧。', '',
        '因此暂不继续增加这版参照选择模块。后续需要区分共同训练得到的表征/分类器差异，以及分别选轮带来的差异；'
        '这只是待检验的问题，不能把训练干扰当成本轮已经证实的原因。', '',
        f"本轮直接重放 C 的三种子十折 30 个原检查点，共 {total} 次重复患者评估、598 位不同患者。"
        '固定所有模型参数和原选轮结果，不重新训练、不重新选轮。先前内部小教师的选择空间不等于这些 C 检查点的选择空间。', '',
        f"重点是无图模型正确、C 错误的 {target['evaluations']} 次评估，涉及 {target['distinct_patient_rows']} 位不同患者。"
        '这组差异还包含两模型重新训练和分别选轮的影响，不能直接视为邻居造成的错误。', '',
        '## 重点病例的变化', '',
        f"- 关闭全部跨患者内容：纠正 {tc['all_closed']['recovered']}/{target['evaluations']} 次。",
        f"- 跨患者内容统一减半：纠正 {tc['constant_half']['recovered']}/{target['evaluations']} 次。",
        f"- 穷举 64 个模态参照开关，只要有一个能判对就计为可恢复：{tr['any_binary']}/{target['evaluations']} 次；"
        f"其中 {tr['binary_but_not_all_closed']} 次不能靠全部关闭纠正。",
        f"- 跳过整个图残差更新、保留 C 的归一化和分类器：纠正 {tc['bypass_graph_update']['recovered']}/{target['evaluations']} 次。",
        f"- 在保持原参照消息强度时，用均值或三次置乱内容替换：至少一种能纠正 {tr['norm_matched_content_any']} 次；"
        f"三次置乱都能纠正的有 {tr['norm_matched_permutation_all_three']} 次。",
        f"- 在本轮全部干预中，仍没有一种能纠正的为 {tr['none_of_tested']}/{target['evaluations']} 次。", '',
        '## 对原本正确预测的代价', '']
    for name, title in [('all_closed','关闭全部跨患者内容'), ('constant_half','统一减半'),
                        ('bypass_graph_update','跳过整个图更新'), ('cohort_mean_norm_matched','强度匹配的均值内容替换')]:
        r = all_summary['conditions'][name]
        text.append(f"- {title}：纠正原 C 的 {r['recovered']} 次错误，同时把 {r['harmed']} 次原本正确的预测改错。")
    text += ['', '## 64 种组合的诊断上限', '',
        f"全部 {total} 次重复评估中，原 C 正确 {oracle['all_open']['correct']} 次；"
        f"所有检查点共用按 ACC 选出的一个最佳组合，正确 {oracle['global_acc_oracle']['correct']} 次；"
        f"逐患者带答案选择的理想上限为 {oracle['individual_acc_oracle']['correct']} 次。", '',
        f"C 的 {all_summary['base_errors']} 次错误中，64 种组合能覆盖的可纠正错误共 "
        f"{all_summary['diagnostic_recovery_sets']['any_binary']} 次；其中仅 "
        f"{all_summary['diagnostic_recovery_sets']['binary_but_not_all_closed']} 次需要部分模态开关、无法通过全部关闭纠正。"
        '这里的纠错覆盖不包含统一关闭参照会造成的错误，因此不能直接作为净提升。', '',
        '**后两者均使用评估标签选择组合，只是探索性诊断，不是新模型的泛化成绩。**'
        '逐患者有一种干预可恢复，不等于有不看答案的规则可以识别并修复这些患者。', '',
        '## 操作与解释边界', '',
        '内容替换时，图、边权、发送/接收变换及原本自身对角贡献保持不变；替换的是外来值通过固定算子形成的消息。'
        '强度匹配版本保持每位接收者、每种内容模态的消息 L2 范数，但不保持方向。'
        '这可以区分部分内容方向敏感性与幅度变化，不能证明哪个具体邻居或哪个医学特征有错。', '',
        '关闭外来内容仍保留图相关的自身对角项；跳过图更新仍保留 C 训练所得参数。因此两者都不是从头训练的无图模型。'
        '这些是训练后干预，存在训练/推断条件变化；未能恢复也不能证明错误只来自自身表征。', '',
        '原检查点采用历史最高测试 ACC 选轮，本轮用同一测试标签作事后描述。没有训练或部署选择器，'
        '没有改变原实验协议，但也不构成独立验证。三个种子复用同一批患者。', '',
        '检查包括原代码/数据/图散列、30 个保存预测重放、无图与 C 行/标签对齐、模型参数保持不变、'
        '原值替换的恒等性、关闭外来值与原门控路径一致性。3 项新增干预实现检查通过。', '',
        '完整数据见 `analysis.json`、`PLAN.json`、`audit.json`、`逐患者干预.csv` 和 `无图对而C错的病例.csv`。']
    (out/'结果分析.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(dict(target=target, all=all_summary, seed_targets={k:v['target'] for k,v in by_seed.items()},
        oracle={k:oracle[k] for k in ('all_open','global_acc_oracle','individual_acc_oracle','prediction_changes')}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
