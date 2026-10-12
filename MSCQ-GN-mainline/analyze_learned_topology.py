"""Replay paired checkpoints and distinguish scoring from topology effects."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from relation_experiment import ROOT,dump,sha,timestamp
from reference_selection_experiment import read,model_hash
from learned_topology_pilot import OUT,SEEDS,MODES,EPOCHS,K,verified_setup,make_model


def main():
    assert read(OUT/'status.json')['state']=='complete'
    plan=read(OUT/'PLAN.json')
    for source,digest in plan['code_sha256'].items():assert sha(ROOT/source)==digest,source
    assert plan['original_protocol_sha256']==sha(ROOT/'Result/reference_selection/PROTOCOL.json')
    cfg,dd,data,graphs,original=verified_setup()
    test=data['Mask'][0][1]
    rows=[];pairs=[];swaps=[];topologies=[];original_c=[]
    with torch.no_grad():
        for seed in SEEDS:
            init=read(OUT/f'initialization_seed_{seed}.json')
            assert init['arms'][0]['model_sha256']==init['arms'][1]['model_sha256']
            predictions={}
            schedules={}
            for mode in MODES:
                folder=OUT/f'seed_{seed}'/mode
                record=read(folder/'complete.json')
                with (folder/'epochs.csv').open(newline='',encoding='utf-8') as stream:
                    history=list(csv.DictReader(stream))
                assert [int(r['epoch']) for r in history]==list(range(1,EPOCHS+1))
                schedules[mode]=[r['lr'] for r in history]
                assert np.isfinite(np.array([[float(r[c]) for c in ('train_loss','test_nll','blend_grad_norm','query_grad_norm')]
                                            for r in history])).all()
                acc=np.array([float(r['test_acc']) for r in history])
                assert int(acc.argmax())+1==record['best']['epoch']
                assert acc[-1]==record['last']['acc']
                assert record['initial_model_sha256']==init['arms'][0]['model_sha256']
                for which in ('best','last'):
                    checkpoint=folder/f'{which}.pt'
                    assert sha(checkpoint)==record[f'{which}_checkpoint_sha256']
                    model=make_model(dd,cfg,data,graphs,seed,mode).eval()
                    model.load_state_dict(torch.load(checkpoint,map_location=cfg.Device,weights_only=False)['state_dict'])
                    logits=model(data['Feature'])[0]
                    filename='predictions.npz' if which=='best' else 'last_predictions.npz'
                    with np.load(folder/filename) as saved:
                        np.testing.assert_allclose(logits[test].cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
                        np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
                        np.testing.assert_array_equal(saved['source_row'],np.asarray(dd['Index'])[test.cpu().numpy()])
                        np.testing.assert_array_equal(saved['neighbors'],model.propagation.last_neighbors.cpu().numpy())
                        np.testing.assert_array_equal(saved['support'],model.propagation.last_support.cpu().numpy())
                        assert np.mean(saved['logits'].argmax(-1)==saved['labels'])==record[which]['acc']
                    topology=model.propagation.topology_diagnostics()
                    assert topology['symmetric'] and topology['no_self_edges']
                    assert all(t['outgoing_k']==K for t in topology['modalities'])
                    topologies.append(dict(seed=seed,mode=mode,checkpoint=which,**topology))
                    correct=logits[test].argmax(-1)==data['Label'][test]
                    predictions[(mode,which)]=correct.cpu().numpy()
                    if mode=='learned':
                        before=model_hash(model)
                        model.propagation.topology_mode='fixed'
                        changed=model(data['Feature'])[0]
                        changed_correct=changed[test].argmax(-1)==data['Label'][test]
                        assert before==model_hash(model)
                        swaps.append(dict(seed=seed,checkpoint=which,learned_acc=float(correct.double().mean()),
                            original_support_acc=float(changed_correct.double().mean()),
                            predictions_changed=int((changed[test].argmax(-1)!=logits[test].argmax(-1)).sum()),
                            learned_corrects=int((~changed_correct&correct).sum()),
                            learned_harms=int((changed_correct&~correct).sum())))
                    del model,logits
                last_topology=record['last_topology']['modalities']
                best_topology=read(folder/'best_topology.json')['modalities']
                rows.append(dict(seed=seed,mode=mode,best_epoch=record['best']['epoch'],best_acc=record['best']['acc'],
                    last_acc=record['last']['acc'],last100_acc=record['mean_last100_acc'],
                    best_new_choice_fraction=float(np.mean([t['new_choice_fraction'] for t in best_topology])),
                    last_new_choice_fraction=float(np.mean([t['new_choice_fraction'] for t in last_topology])),
                    best_mean_degree=float(np.mean([t['degree_mean'] for t in best_topology])),
                    last_mean_degree=float(np.mean([t['degree_mean'] for t in last_topology])),parameters=record['parameters']))
            assert schedules['fixed']==schedules['learned']
            for which in ('best','last'):
                a,b=predictions[('fixed',which)],predictions[('learned',which)]
                pairs.append(dict(seed=seed,checkpoint=which,delta_acc_pp=100*float(b.mean()-a.mean()),
                    corrected=int((~a&b).sum()),harmed=int((a&~b).sum())))
            historical=read(ROOT/f'Result/relation_mechanism_nodecay/seed_{seed}/content/fold_01/complete.json')
            original_c.append(dict(seed=seed,best_acc=historical['best']['acc'],last_acc=historical['last']['acc']))
    averages={mode:{key:float(np.mean([r[key] for r in rows if r['mode']==mode])) for key in
        ('best_acc','last_acc','last100_acc','best_new_choice_fraction','last_new_choice_fraction','best_mean_degree','last_mean_degree')}
        for mode in MODES}
    delta=100*(averages['learned']['best_acc']-averages['fixed']['best_acc'])
    last_delta=100*(averages['learned']['last_acc']-averages['fixed']['last_acc'])
    consistent=all(r['delta_acc_pp']>0 for r in pairs if r['checkpoint']=='best')
    if consistent:
        verdict='三个种子的原选轮 ACC 均提升，值得在原十折上进一步验证；单折结果尚不能确认稳定泛化收益。'
    elif delta>1e-9:
        verdict='原选轮 ACC 均值提高，但种子间不是一致提升，暂不能认定稳定有效。'
    elif abs(delta)<1e-9:
        verdict='原选轮 ACC 均值没有提升，暂未证明可学习邻居优于同等打分器的固定图。'
    else:
        verdict='原选轮 ACC 均值下降，当前这版可学习邻居未通过首轮筛查。'
    historical_mean=float(np.mean([r['best_acc'] for r in original_c]))
    historical_delta=100*(averages['learned']['best_acc']-historical_mean)
    if historical_delta < -1e-9:
        verdict+=' 可学习组仍低于历史原 C，不能作为优于原主线的新模型。'
    elif abs(historical_delta)<1e-9:
        verdict+=' 可学习组与历史原 C 的均值持平，尚未超过原主线。'
    analysis=dict(completed_at=timestamp(),rows=rows,averages=averages,pairs=pairs,topologies=topologies,
        checkpoint_support_swaps=swaps,original_c_context=original_c,verdict=verdict,
        learned_vs_historical_c_best_delta_pp=historical_delta,
        scope='Original fold 1, same 60 test patients repeated at three training seeds; 1000 epochs from scratch',
        limitations=['Historical TEST-ACC selection; exploratory',
            'Fixed control adds the same six score-blend parameters, so differs from exact historical C',
            'Outgoing k=10 matched but reverse-union degrees can differ',
            'One bounded cosine blend and one straight-through temperature; not a test of every topology learner',
            'Checkpoint support swaps are post-training distribution changes, not replacement training results'],
        analysis_source_sha256=sha(Path(__file__)))
    dump(OUT/'analysis.json',analysis)
    dump(OUT/'audit.json',dict(passed=True,paired_runs=6,replayed_checkpoints=12,
        original_sources_data_splits_unchanged=True,first_test_maxima_verified=True,
        original_initialization_verified=True,paired_initializations_identical=True,
        equal_parameter_counts=True,identical_lr_schedules=True,forward_topology_and_predictions_replayed=True))
    with (OUT/'逐组结果.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=dict(fixed='固定邻居＋同等关系打分器',learned='可学习邻居＋同等关系打分器')
    lines=['# C 的端到端可学习邻居：首轮筛查','',f'**{verdict}**','',
        f'可学习邻居相对固定图：原选轮 ACC 差值 {delta:+.2f} 个百分点，固定第 1000 轮差值 {last_delta:+.2f} 个百分点。','',
        '## 实验怎么做','',
        'TADPOLE AD/CN/SMCI，原第 1 折、训练 seed 0/1/2，同一批 60 位测试患者。每组从随机初始化训练 1000 轮，'
        '没有预训练、冻结、蒸馏或分阶段接图。保留原 C 内容模态变换、两次协同传播、分类器、损失、噪声和优化设置。','',
        '两组都复用 C 的关系查询投影，并各加入相同的 6 个模态混合系数，构图分数为原始余弦相似度加可学习系数乘编码表示相似度。'
        '混合系数从零开始，两组初始图、共同参数及预测相同。固定组只用该分数调整原边权；可学习组还用该分数从全体其他患者中更新实际邻居。'
        '因此主对照是同等打分能力下能否换邻居，不把多加的打分参数算作拓扑增益。','',
        '每位患者在每个关系模态中先选 10 个近邻，再取反向边并集以符合 C 的对称算子。训练前向和评估都使用实际选中的图，'
        '反向使用固定总量的连续选择近似，给未选中边提供梯度。这是可微选择启发的简化实现，使用有偏直通梯度，不是原 DGG 的完整复现。'
        '两组使用相同的稠密批量计算内核；原稀疏 C 的传播值和梯度一致性已测试。','',
        '主指标沿用全部 1000 轮第一次最高测试 ACC，同时列出末轮和末 100 轮。单折三种子不是十折结果，也不是三批独立患者。','',
        '## 结果','']
    for mode in MODES:
        a=averages[mode]
        lines.append(f"- {names[mode]}：原选轮平均 ACC {100*a['best_acc']:.2f}%，末轮 {100*a['last_acc']:.2f}%，末 100 轮均值 {100*a['last100_acc']:.2f}%。")
    for seed in SEEDS:
        a=next(r for r in rows if r['seed']==seed and r['mode']=='fixed')
        b=next(r for r in rows if r['seed']==seed and r['mode']=='learned')
        lines.append(f"- seed {seed}：固定图 {100*a['best_acc']:.2f}%（第 {a['best_epoch']} 轮），可学习图 {100*b['best_acc']:.2f}%（第 {b['best_epoch']} 轮）；"
            f"末轮分别为 {100*a['last_acc']:.2f}%、{100*b['last_acc']:.2f}%。")
    lines+=['',f"历史原 C 在同一折三个种子的选轮均值为 {100*np.mean([r['best_acc'] for r in original_c]):.2f}%，"
        '仅作为已有模型水平的参照；上述固定组因加入相同的关系打分系数而不等于原 C。','',
        f"同等打分器的固定组比历史原 C 低 {100*(historical_mean-averages['fixed']['best_acc']):.2f} 个百分点；"
        f"允许换邻居后改善 {delta:.2f} 个百分点，但相对原 C 仍为 {historical_delta:+.2f} 个百分点。"
        '因此不能只引用本轮匹配对照的提升就声称超过原主线。','',
        '## 邻居是否真的改变','',
        '以下比例按全体 598 位患者、六个模态等权平均；不是只统计测试患者。新的选择指原有向 top-10 列表之外的患者。','']
    for r in rows:
        if r['mode']!='learned':continue
        lines.append(f"- seed {r['seed']}：最佳轮新邻居选择比例 {100*r['best_new_choice_fraction']:.2f}%，末轮 {100*r['last_new_choice_fraction']:.2f}%；"
            f"对称化后的平均度最佳轮 {r['best_mean_degree']:.2f}、末轮 {r['last_mean_degree']:.2f}。")
    lines+=['',f"固定组对称化后平均度为 {averages['fixed']['best_mean_degree']:.2f}。两组匹配的是对称化前的选择预算；"
        '如性能提高仍须注意实际度数变化这一因素。图变化或同质性变化本身不能证明选择更有帮助。','',
        '## 同一检查点换回原候选图','',
        '固定可学习模型的全部参数，仅把推断时的邻居支持换回原图。这里只诊断模型对所学拓扑的依赖，不能替代重新训练对照。','']
    for r in swaps:
        lines.append(f"- seed {r['seed']}，{'最佳轮' if r['checkpoint']=='best' else '末轮'}：所学图 {100*r['learned_acc']:.2f}%，"
            f"换回原图 {100*r['original_support_acc']:.2f}%；所学图相对原图纠正 {r['learned_corrects']} 次、改错 {r['learned_harms']} 次。")
    lines+=['','## 已完成检查与解释范围','',
        '5 项实现测试通过，包括原算子值/梯度一致、连续选择约束与数值梯度、未选中边的训练信号、初始化和检查点重放。'
        '完整数据上的初始预测与原 C 一致；两组参数规模、起点和学习率进度一致。12 个最佳/末轮检查点、预测及邻居列表已重放。','',
        '本轮只筛查一种原始相似度与编码相似度混合的关系分数，固定选择温度 0.1、预算 10。'
        '负结果不能推广为所有端到端构图无效；正结果也不能直接证明医学关系正确或形成新贡献。原工程和既有实验结果未覆盖。','',
        '## 下一步建议','',
        '先保留原 C 主线和本轮结果，不立即叠加自适应邻域大小、互补约束等模块。'
        '下一轮可让新关系分数只负责选择邻居，同时恢复 C 原有的边权计算，并配置相同计算内核的原 C 对照，'
        '进一步区分候选拓扑变化与边权打分改动的影响。该建议尚未执行，也不能据当前结果断定新增打分一定是性能损失的唯一原因。']
    (OUT/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verdict=verdict,averages=averages,rows=rows,pairs=pairs,swaps=swaps),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
