"""Replay paired checkpoints and distinguish scoring from topology effects."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from relation_experiment import ROOT,dump,sha,timestamp,scores
from reference_selection_experiment import read,model_hash
from Model.relation_sheaf import RelationSheafModel
from Utils import SET_Random
from selection_only_pilot import OUT,SEEDS,MODES,EPOCHS,K,verified_setup,make_model


def main():
    assert read(OUT/'status.json')['state']=='complete'
    plan=read(OUT/'PLAN.json')
    for source,digest in plan['code_sha256'].items():assert sha(ROOT/source)==digest,source
    assert plan['original_protocol_sha256']==sha(ROOT/'Result/reference_selection/PROTOCOL.json')
    cfg,dd,data,graphs,original=verified_setup()
    test=data['Mask'][0][1]
    rows=[];pairs=[];swaps=[];topologies=[];original_c=[];sparse_checks=[];class_metrics=[];reproduction_trace=[]
    with torch.no_grad():
        for seed in SEEDS:
            init=read(OUT/f'initialization_seed_{seed}.json')
            assert init['arms'][0]['common_sha256']==init['arms'][1]['common_sha256']
            assert init['original_cpu_cuda_rng_preserved']
            assert [a['parameters'] for a in init['arms']]==[895096,895102]
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
                selector_grad=np.array([float(r['blend_grad_norm']) for r in history])
                assert (selector_grad>0).any() if mode=='learned' else (selector_grad==0).all()
                acc=np.array([float(r['test_acc']) for r in history])
                assert int(acc.argmax())+1==record['best']['epoch']
                assert acc[-1]==record['last']['acc']
                assert record['initial_model_sha256']==next(a['model_sha256'] for a in init['arms'] if a['mode']==mode)
                assert record['parameters']==(895096 if mode=='fixed' else 895102)
                assert np.isclose(acc[-100:].mean(),record['mean_last100_acc'])
                assert np.isclose(acc.max(),record['best']['acc'])
                if mode=='fixed':
                    historical_csv=ROOT/f'Result/relation_mechanism_nodecay/seed_{seed}/content/fold_01/epochs.csv'
                    with historical_csv.open(newline='',encoding='utf-8') as stream:
                        historical_history=list(csv.DictReader(stream))
                    for epoch in (1,2,3,10,20,100):
                        old,new=historical_history[epoch-1],history[epoch-1]
                        assert old['lr']==new['lr']
                        reproduction_trace.append(dict(seed=seed,epoch=epoch,
                            sparse_train_loss=float(old['train_loss']),dense_train_loss=float(new['train_loss']),
                            loss_difference=abs(float(old['train_loss'])-float(new['train_loss'])),
                            sparse_acc=float(old['test_acc']),dense_acc=float(new['test_acc'])))
                for which in ('best','last'):
                    checkpoint=folder/f'{which}.pt'
                    assert sha(checkpoint)==record[f'{which}_checkpoint_sha256']
                    model=make_model(dd,cfg,data,graphs,seed,mode).eval()
                    model.load_state_dict(torch.load(checkpoint,map_location=cfg.Device,weights_only=False)['state_dict'])
                    logits=model(data['Feature'])[0]
                    for name,value in scores(logits[test],data['Label'][test]).items():
                        np.testing.assert_allclose(value,record[which][name],atol=1e-8,rtol=1e-8)
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
                    # At trained checkpoints, fixed support must still equal C.
                    SET_Random(seed)
                    sparse=RelationSheafModel(dd,cfg,graphs,'content',training_seed=seed).to(cfg.Device).eval()
                    sparse.load_state_dict({name:value for name,value in model.state_dict().items()
                                            if name!='propagation.relation_blend'},strict=True)
                    sparse_logits=sparse(data['Feature'])[0]
                    dense_logits=changed if mode=='learned' else logits
                    max_error=float((dense_logits-sparse_logits).abs().max())
                    torch.testing.assert_close(dense_logits,sparse_logits,atol=2e-5,rtol=2e-5)
                    assert torch.equal(dense_logits.argmax(-1),sparse_logits.argmax(-1))
                    sparse_checks.append(dict(seed=seed,mode=mode,checkpoint=which,max_abs_error=max_error))
                    class_metrics.append(dict(seed=seed,mode=mode,checkpoint=which,
                        recalls=record[which]['per_class_recall'],macro_f1=record[which]['macro_f1'],
                        confusion_matrix=record[which]['confusion_matrix']))
                    del model,logits,sparse,sparse_logits
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
        verdict='原选轮 ACC 均值没有提升，暂未证明可学习邻居优于原 C 固定图。'
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
        trained_sparse_c_replays=sparse_checks,class_metrics=class_metrics,
        historical_sparse_vs_dense_early_trace=reproduction_trace,
        learned_vs_historical_c_best_delta_pp=historical_delta,
        scope='Original fold 1, same 60 test patients repeated at three training seeds; 1000 epochs from scratch',
        limitations=['Historical TEST-ACC selection; exploratory',
            'Fixed control is C with a dense equivalent kernel; historical sparse training trajectories can differ in floating point',
            'Outgoing k=10 matched but reverse-union degrees can differ',
            'One bounded cosine blend and one straight-through temperature; not a test of every topology learner',
            'Checkpoint support swaps are post-training distribution changes, not replacement training results'],
        analysis_source_sha256=sha(Path(__file__)))
    dump(OUT/'analysis.json',analysis)
    dump(OUT/'audit.json',dict(passed=True,paired_runs=6,replayed_checkpoints=12,
        original_sources_data_splits_unchanged=True,first_test_maxima_verified=True,
        original_initialization_verified=True,paired_common_initializations_identical=True,
        fixed_parameters=895096,learned_parameters=895102,identical_lr_schedules=True,
        forward_topology_and_predictions_replayed=True,original_cpu_cuda_rng_preserved=True,
        trained_sparse_c_equivalence_checks=sparse_checks))
    with (OUT/'逐组结果.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_report(analysis)
    print(json.dumps(dict(verdict=verdict,averages=averages,rows=rows,pairs=pairs,swaps=swaps),ensure_ascii=False,indent=2))


def write_report(a):
    means=a['averages'];rows=a['rows'];pairs=a['pairs'];swaps=a['checkpoint_support_swaps']
    delta=100*(means['learned']['best_acc']-means['fixed']['best_acc'])
    last_delta=100*(means['learned']['last_acc']-means['fixed']['last_acc'])
    names=dict(fixed='原 C 固定图（同一计算内核）',learned='只学习邻居，保留 C 边权公式')
    lines=['# 保留 C 边权，仅学习邻居：首轮结果','',f"**{a['verdict']}**",'',
        f'相对本轮原 C 对照，原选轮 ACC 变化 {delta:+.2f} 个百分点；固定第 1000 轮变化 {last_delta:+.2f} 个百分点。','',
        '## 本轮验证什么','',
        '上一轮同时使用新分数选邻居和调整边权，无法直接判断相对原 C 的变化来自哪部分。本轮恢复 C 原来的边权公式，'
        '新增分数只用于选择邻居，检验这个端到端选择器是否能改进原 C。两组均从头联合训练，未冻结任何模块。','',
        '固定组保留原始各模态余弦 top-10 图，参数量 895096；可学习组复用现有查询表示，仅新增 6 个选择系数，参数量 895102。'
        '选择分数 = 原始模态余弦 + tanh(系数) × 对应模态查询表示余弦；每行选 10 人，再加反向边并集。'
        '边权始终为 sigmoid(内容模态查询余弦 + 可学习系数 × 建边模态原始余弦 + 偏置)。新增的选择系数不进入边权公式。','',
        '编码器、查询投影、内容变换和分类器仍联合更新，因此训练后共同参数可以不同；“保留边权”指保留计算公式，'
        '不是冻结数值。选择器用硬 top-10 前向与固定总量连续近似的直通反向，这是有偏代理梯度。','',
        '## 协议','',
        'TADPOLE AD/CN/SMCI，沿用原十折划分中的第 1 折：538 位训练患者、60 位测试患者。'
        '每组训练 seed 0/1/2、1000 轮；使用原预处理、损失、Adam、学习率进度、输入噪声与正则化配置。'
        '原 C 和新组都使用相同的稠密计算内核；固定组与原稀疏 C 的传播值和梯度已校验。','',
        '选轮仍按历史协议取 1000 轮中第一次最高测试 ACC，另外报告固定第 1000 轮与末 100 轮均值。'
        '这是单折初筛，三个种子对应同一批测试患者，不是十折均值或三批独立患者。','',
        '## 结果','']
    for mode in MODES:
        m=means[mode]
        lines.append(f"- {names[mode]}：选轮 ACC {100*m['best_acc']:.2f}%，末轮 {100*m['last_acc']:.2f}%，末 100 轮均值 {100*m['last100_acc']:.2f}%。")
    for seed in SEEDS:
        fixed=next(r for r in rows if r['seed']==seed and r['mode']=='fixed')
        learned=next(r for r in rows if r['seed']==seed and r['mode']=='learned')
        lines.append(f"- seed {seed}：原 C {100*fixed['best_acc']:.2f}%（第 {fixed['best_epoch']} 轮），新组 {100*learned['best_acc']:.2f}%（第 {learned['best_epoch']} 轮）；"
            f"末轮分别 {100*fixed['last_acc']:.2f}%、{100*learned['last_acc']:.2f}%。")
    for which,label in (('best','选轮'),('last','末轮')):
        p=[r for r in pairs if r['checkpoint']==which]
        lines.append(f"- {label}逐患者配对：合计纠正 {sum(r['corrected'] for r in p)} 次，新增错误 {sum(r['harmed'] for r in p)} 次。计数跨三个重复种子，不等于独立患者数。")
    lines+=['','## 类别表现','']
    for mode in MODES:
        metrics=[r for r in a['class_metrics'] if r['mode']==mode and r['checkpoint']=='best']
        recall=np.mean([r['recalls'] for r in metrics],axis=0)*100
        macro=np.mean([r['macro_f1'] for r in metrics])*100
        lines.append(f"- {names[mode]}：选轮平均召回率 AD {recall[0]:.2f}%、CN {recall[1]:.2f}%、SMCI {recall[2]:.2f}%；宏 F1 {macro:.2f}%。")
    lines+=['','## 是否真的选了新邻居','',
        '以下按全体 598 位患者和六个模态等权统计；新选择是原有向 top-10 列表之外的患者。','']
    for r in rows:
        if r['mode']!='learned':continue
        lines.append(f"- seed {r['seed']}：选轮新邻居比例 {100*r['best_new_choice_fraction']:.2f}%，末轮 {100*r['last_new_choice_fraction']:.2f}%；"
            f"对称化后平均度分别 {r['best_mean_degree']:.2f}、{r['last_mean_degree']:.2f}。")
    modal_names=read(ROOT/'Result/reference_selection/PROTOCOL.json')['modalities']
    learned_topologies=[r for r in a['topologies'] if r['mode']=='learned' and r['checkpoint']=='best']
    fixed_topology=next(r for r in a['topologies'] if r['mode']=='fixed')
    lines+=['','各模态选轮图（三种子）：','']
    for index,name in enumerate(modal_names):
        stats=[r['modalities'][index] for r in learned_topologies]
        fraction=100*np.mean([r['new_choice_fraction'] for r in stats])
        degree_maxima=[r['degree_max'] for r in stats]
        lines.append(f"- {name}：平均新选择 {fraction:.2f}%；最大度原图 {fixed_topology['modalities'][index]['degree_max']}，"
            f"新图三个种子分别 {degree_maxima}。")
    lines += ['',f"原 C 平均度 {means['fixed']['best_mean_degree']:.2f}；匹配的是对称化之前每人每模态选 10 人的预算，"
        '反向边并集之后的度数仍可能不同。图改变本身不代表更有预测价值。','',
        '## 固定检查点，换回原图','',
        '固定新组的全部训练参数，只在推断时换回原邻居，边权公式保持不变。这是依赖性诊断，不能替代重新训练对照。','']
    for r in swaps:
        lines.append(f"- seed {r['seed']}，{'选轮' if r['checkpoint']=='best' else '末轮'}：所学图 {100*r['learned_acc']:.2f}% → 原图 {100*r['original_support_acc']:.2f}%；"
            f"所学图相对原图纠正 {r['learned_corrects']} 次、改错 {r['learned_harms']} 次。")
    for which,label in (('best','选轮'),('last','末轮')):
        records=[r for r in swaps if r['checkpoint']==which]
        learned_mean=100*np.mean([r['learned_acc'] for r in records])
        swapped_mean=100*np.mean([r['original_support_acc'] for r in records])
        lines.append(f'- {label}三种子平均：所学图 {learned_mean:.2f}% → 原图 {swapped_mean:.2f}%。')
    historical=np.mean([r['best_acc'] for r in a['original_c_context']])*100
    previous=read(ROOT/'Result/learned_topology_pilot/analysis.json')['averages']
    lines += ['','## 与已有记录的关系','',
        f"历史稀疏 C 同折三种子选轮均值 {historical:.2f}%；本轮同公式稠密 C 为 {100*means['fixed']['best_acc']:.2f}%。"
        '两种内核在固定权重下数值等价，但浮点求和差异可能在长程训练中累积，不能要求重新训练后每轮指标相同。'
        '本轮主要依据相同内核的配对对照判断，而非跨次历史最高值。','',
        f"上一轮同时使用新边权的固定组 / 换邻居组分别为 {100*previous['fixed']['best_acc']:.2f}% / {100*previous['learned']['best_acc']:.2f}%。"
        '本次复核发现，上一轮新增模块的 CPU-only 随机数保护改变了 CUDA 随机序列。上一轮两组之间序列相同，'
        '但与历史 C 不同，不能把其固定组低于历史 C 全部归因于改了边权。本轮已保留原 C 的 CPU 与所有 CUDA 随机状态。'
        '跨轮同时涉及随机序列修正，不能把跨轮差值视为单独去掉新边权的因果效应。','',
        '本次修正前已启动的一轮位于 Result/selection_only_pilot，发现问题后中止并标记 SUPERSEDED；'
        '未把其中的结果纳入本报告。正式结果全部来自修正后的 Result/selection_only_rng_pilot。','',
        '## 已完成检查','',
        '10 项实现测试通过：原算子值与梯度一致、固定基线所有梯度一致、选择系数不直接改变边权、未选中边可获得代理梯度、'
        '初始化与随机数流一致、检查点重放等。完整数据初始预测、共同参数和图一致；固定组没有额外闲置参数。'
        '额外验证了 GPU 初始化前后及训练前向结束时的随机状态与原 C 一致。'
        '已核对原文件与数据哈希、6 组完整训练日志和学习率、12 个保存检查点与预测/邻居列表，并在训练后权重上再次验证稀疏 C 等价性。','',
        '当前实验只考察固定预算 10、温度 0.1、原始余弦与查询余弦混合的一种选择器；不能据此判断所有图学习方案。'
        '训练损失保持原掩码；图使用全队列特征，沿用原传导式设置。测试标签参与历史选轮，因此结果用于探索，不作为独立泛化证明。']
    (OUT/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
