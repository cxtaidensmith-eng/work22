"""Audit paired graph-supervision experiments and the cached original C control."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from relation_experiment import ROOT,dump,sha,timestamp,scores
from reference_selection_experiment import read,model_hash
from Model.relation_sheaf import RelationSheafModel
from Model.task_guided_graph import balanced_relation_loss
from selection_only_pilot import make_model as make_c_model
from task_guided_graph_pilot import (OUT,BASELINE,SEEDS,MODES,EPOCHS,K,RELATION_WEIGHT,
                                    verified_setup,verify_baseline,make_model)

NAMES={'c':'原 C','head':'独立构图头','supervised':'独立构图头＋关系监督'}


def relation_diagnostics(score,support,labels,fit):
    indices=fit.nonzero().flatten()
    y=labels[indices]
    same=y[:,None]==y[None,:]
    upper=torch.ones_like(same).triu(1)
    train_support=support[:,indices][:,:,indices]&upper
    train_score=score[:,indices][:,:,indices]
    target=same[upper].cpu().numpy()
    auc=[roc_auc_score(target,s[upper].cpu().numpy()) for s in train_score]
    homophily=[float((s&same).sum()/s.sum()) for s in train_support]
    loss=float(balanced_relation_loss(score,labels,fit))
    per_modality=[float(balanced_relation_loss(score[m:m+1],labels,fit)) for m in range(len(score))]
    masked=labels.clone();masked[~fit]=99999
    assert float(balanced_relation_loss(score,masked,fit))==loss
    return dict(fit_relation_mse=loss,fit_relation_mse_per_modality=per_modality,
                fit_score_auc=auc,fit_edge_homophily=homophily,
                scope='Unordered FIT-FIT pairs only, no diagonal; descriptive, not held-out graph validation')


def main():
    assert read(OUT/'status.json')['state']=='complete'
    plan=read(OUT/'PLAN.json')
    for source,digest in plan['code_sha256'].items():assert sha(ROOT/source)==digest,source
    assert verify_baseline()==plan['baseline_manifest']
    assert plan['original_protocol_sha256']==sha(ROOT/'Result/reference_selection/PROTOCOL.json')
    cfg,dd,data,graphs,_=verified_setup()
    fit,test=data['Mask'][0]
    rows=[];pairs=[];swaps=[];topologies=[];class_metrics=[];sparse_checks=[]
    with torch.no_grad():
        for seed in SEEDS:
            init=read(OUT/f'initialization_seed_{seed}.json')
            cached_init=read(BASELINE/f'initialization_seed_{seed}.json')
            assert init['original_cpu_cuda_rng_preserved']
            assert init['original_cuda_rng_sha256']==cached_init['original_cuda_rng_sha256']
            assert init['arms'][0]['model_sha256']==init['arms'][1]['model_sha256']
            predictions={};schedules={}
            for mode in ('c',*MODES):
                folder=(BASELINE/f'seed_{seed}/fixed' if mode=='c' else OUT/f'seed_{seed}/{mode}')
                record=read(folder/'complete.json')
                history=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
                assert [int(r['epoch']) for r in history]==list(range(1,EPOCHS+1))
                schedules[mode]=[r['lr'] for r in history]
                acc=np.array([float(r['test_acc']) for r in history])
                assert int(acc.argmax())+1==record['best']['epoch']
                assert acc.max()==record['best']['acc'] and acc[-1]==record['last']['acc']
                assert np.isclose(acc[-100:].mean(),record['mean_last100_acc'])
                expected_params=895096 if mode=='c' else 897022
                assert record['parameters']==expected_params
                if mode!='c':
                    assert record['initial_model_sha256']==init['arms'][0]['model_sha256']
                    fields=('train_loss','task_loss','relation_loss','test_nll','blend_grad_norm',
                            'query_grad_norm','graph_query_grad_norm','metric_grad_norm')
                    values={key:np.array([float(r[key]) for r in history]) for key in fields}
                    assert all(np.isfinite(value).all() for value in values.values())
                    coefficient=RELATION_WEIGHT if mode=='supervised' else 0.
                    np.testing.assert_allclose(values['train_loss'],values['task_loss']+coefficient*values['relation_loss'],atol=1e-5,rtol=1e-6)
                    assert (values['graph_query_grad_norm']>0).any() and (values['metric_grad_norm']>0).any()
                diagnostics={}
                for which in ('best','last'):
                    checkpoint=folder/f'{which}.pt'
                    assert sha(checkpoint)==record[f'{which}_checkpoint_sha256']
                    model=(make_c_model(dd,cfg,data,graphs,seed,'fixed') if mode=='c'
                           else make_model(dd,cfg,data,graphs,seed,mode)).eval()
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
                    topology=model.propagation.topology_diagnostics()
                    assert topology['symmetric'] and topology['no_self_edges']
                    assert all(t['outgoing_k']==K for t in topology['modalities'])
                    relation_score=(model.propagation.raw_similarity if mode=='c' else model.propagation.last_selection_score)
                    relation=relation_diagnostics(relation_score,model.propagation.last_support,data['Label'],fit)
                    diagnostics[which]=(topology,relation)
                    topologies.append(dict(seed=seed,mode=mode,checkpoint=which,**topology,**relation))
                    correct=logits[test].argmax(-1)==data['Label'][test]
                    predictions[(mode,which)]=correct.cpu().numpy()
                    if mode!='c':
                        before=model_hash(model)
                        model.propagation.topology_mode='fixed'
                        changed=model(data['Feature'])[0]
                        changed_correct=changed[test].argmax(-1)==data['Label'][test]
                        assert before==model_hash(model)
                        swaps.append(dict(seed=seed,mode=mode,checkpoint=which,
                            learned_acc=float(correct.double().mean()),original_support_acc=float(changed_correct.double().mean()),
                            corrected=int((~changed_correct&correct).sum()),harmed=int((changed_correct&~correct).sum())))
                        sparse=RelationSheafModel(dd,cfg,graphs,'content',training_seed=seed).to(cfg.Device).eval()
                        extra=('propagation.relation_blend','propagation.metric_weights','propagation.graph_query.weight')
                        sparse.load_state_dict({k:v for k,v in model.state_dict().items() if k not in extra},strict=True)
                        sparse_logits=sparse(data['Feature'])[0]
                        torch.testing.assert_close(changed,sparse_logits,atol=2e-5,rtol=2e-5)
                        assert torch.equal(changed.argmax(-1),sparse_logits.argmax(-1))
                        sparse_checks.append(dict(seed=seed,mode=mode,checkpoint=which,
                            max_error=float((changed-sparse_logits).abs().max())))
                        del sparse,sparse_logits
                    class_metrics.append(dict(seed=seed,mode=mode,checkpoint=which,
                        recalls=record[which]['per_class_recall'],macro_f1=record[which]['macro_f1']))
                    del model,logits
                best_top,best_rel=diagnostics['best'];last_top,last_rel=diagnostics['last']
                rows.append(dict(seed=seed,mode=mode,best_epoch=record['best']['epoch'],best_acc=record['best']['acc'],
                    last_acc=record['last']['acc'],last100_acc=record['mean_last100_acc'],parameters=record['parameters'],
                    best_fit_relation_mse=best_rel['fit_relation_mse'],last_fit_relation_mse=last_rel['fit_relation_mse'],
                    best_fit_auc=float(np.mean(best_rel['fit_score_auc'])),last_fit_auc=float(np.mean(last_rel['fit_score_auc'])),
                    best_fit_homophily=float(np.mean(best_rel['fit_edge_homophily'])),
                    best_new_choices=float(np.mean([r['new_choice_fraction'] for r in best_top['modalities']])),
                    last_new_choices=float(np.mean([r['new_choice_fraction'] for r in last_top['modalities']])),
                    best_mean_degree=float(np.mean([r['degree_mean'] for r in best_top['modalities']]))))
            assert schedules['c']==schedules['head']==schedules['supervised']
            for base,new in (('c','head'),('head','supervised'),('c','supervised')):
                for which in ('best','last'):
                    left,right=predictions[(base,which)],predictions[(new,which)]
                    pairs.append(dict(seed=seed,base=base,new=new,checkpoint=which,
                        delta_acc_pp=100*float(right.mean()-left.mean()),corrected=int((~left&right).sum()),harmed=int((left&~right).sum())))
    keys=('best_acc','last_acc','last100_acc','best_fit_relation_mse','last_fit_relation_mse',
          'best_fit_auc','last_fit_auc','best_fit_homophily','best_new_choices','last_new_choices','best_mean_degree')
    means={mode:{key:float(np.mean([r[key] for r in rows if r['mode']==mode])) for key in keys}
           for mode in ('c',*MODES)}
    gain_head=100*(means['supervised']['best_acc']-means['head']['best_acc'])
    gain_c=100*(means['supervised']['best_acc']-means['c']['best_acc'])
    if gain_head>1e-9 and gain_c>1e-9:
        verdict='关系监督提高了单折选轮均值，且超过原 C；仍需结合种子一致性和固定末轮判断，不能据此确认泛化收益。'
    elif gain_head>1e-9:
        verdict='关系监督优于相同构图头，但尚未超过原 C；不能作为优于主线的新模型。'
    elif abs(gain_head)<1e-9:
        verdict='关系监督没有提高相同构图头的选轮 ACC 均值，尚未验证新增监督带来收益。'
    else:
        verdict='关系监督降低了相同构图头的选轮 ACC 均值，本轮没有支持增加这项监督。'
    result=dict(verdict=verdict,averages=means,rows=rows,pairs=pairs,swaps=swaps,topologies=topologies,
        class_metrics=class_metrics,supervised_vs_head_best_delta_pp=gain_head,supervised_vs_c_best_delta_pp=gain_c,
        scope='Original fold 1, same 60 test patients, seeds 0/1/2; first maximum TEST ACC; exploratory',
        completed_at=timestamp(),analysis_source_sha256=sha(Path(__file__)))
    dump(OUT/'analysis.json',result)
    dump(OUT/'audit.json',dict(passed=True,new_runs=6,cached_c_runs=3,replayed_checkpoints=18,
        source_data_and_splits_verified=True,baseline_manifest_unchanged=True,original_cpu_cuda_rng_preserved=True,
        same_new_arm_parameters=897022,original_c_parameters=895096,loss_decomposition_verified=True,
        heldout_labels_excluded_from_relation_loss=True,paired_lr_identical=True,trained_sparse_c_checks=sparse_checks))
    with (OUT/'逐组结果.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_report(result)
    print(json.dumps(dict(verdict=verdict,averages=means,rows=rows),ensure_ascii=False,indent=2))


def write_report(result):
    a=result['averages'];rows=result['rows']
    lines=['# 独立构图头与关系监督：单折对照','',f"**{result['verdict']}**",'',
        '## 如何实验','',
        'TADPOLE AD/CN/SMCI，原第 1 折、seed 0/1/2，每组从头端到端训练 1000 轮。原 C 复用此前已审计的同计算内核结果，'
        '核对了文件哈希、数据、划分、共同初始化、CPU/CUDA 随机状态和学习率进度。新增两个版本各训练三个种子。','',
        '独立构图头：增加 96→16 的构图投影，与 C 的边权查询投影分开；每模态用四组可学习维度权重计算加权余弦，再平均。'
        '最终选择分数为原始余弦＋tanh(模态系数)×学习相似度；模态系数从零开始。每模态每人选 10 位邻居后取反向边并集。'
        '保留 C 的边权公式、内容变换、协同传播、分类器和原任务损失。','',
        '关系监督组与独立头组的结构、参数量、初始化完全相同，仅多加 0.1×关系损失。损失直接作用于选邻居使用的最终分数，'
        '目标为训练集内同类患者对 1、异类患者对 0；去掉自配对和重复方向。各同类类别等权，各异类类别组合等权，同类和异类各占一半。'
        '这是软辅助监督，不直接用标签生成预测时的邻接。监督权重预先固定，未按本轮测试表现调参。','',
        '原 C 参数量 895096，两个新版本均为 897022。独立构图头与原 C 的比较包含容量和度量形式变化；'
        '监督组与独立头组的比较只增加关系损失。两组都保留可微选择的近似反向梯度，不需要预训练、冻结或分阶段接图。','',
        '借鉴 [IDGL（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) 的可学习关系度量，'
        '以及 [SPromptGL（MICCAI 2025）](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) 的监督图目标。'
        '本轮是适配 C 的模块验证，不是两篇论文的完整复现。','',
        '## 结果','']
    for mode in ('c',*MODES):
        r=a[mode]
        lines.append(f"- {NAMES[mode]}：原选轮 ACC {100*r['best_acc']:.2f}%，末轮 {100*r['last_acc']:.2f}%，末 100 轮均值 {100*r['last100_acc']:.2f}%。")
    for seed in SEEDS:
        r={mode:next(r for r in rows if r['seed']==seed and r['mode']==mode) for mode in ('c',*MODES)}
        lines.append(f"- seed {seed}：原 C / 独立头 / 关系监督选轮 ACC 分别 {100*r['c']['best_acc']:.2f}% / {100*r['head']['best_acc']:.2f}% / {100*r['supervised']['best_acc']:.2f}%；"
            f"末轮分别 {100*r['c']['last_acc']:.2f}% / {100*r['head']['last_acc']:.2f}% / {100*r['supervised']['last_acc']:.2f}%。")
    for base,new in (('c','head'),('head','supervised'),('c','supervised')):
        matched=[r for r in result['pairs'] if r['base']==base and r['new']==new and r['checkpoint']=='best']
        lines.append(f"- {NAMES[new]}相对{NAMES[base]}：三种子选轮合计纠正 {sum(r['corrected'] for r in matched)} 次、新增错误 {sum(r['harmed'] for r in matched)} 次；"
            f"各种子 ACC 差值 {[round(r['delta_acc_pp'],2) for r in matched]} 个百分点。")
    lines+=['','## 关系监督是否学到了信号','',
        '以下只统计训练集内患者对，不是独立验证集指标。原 C 的关系评分指标以原始构图余弦计算；'
        '两个新组以实际用于选择邻居的分数计算。训练对关系拟合改善不等于分类泛化改善。','']
    for mode in ('c',*MODES):
        r=a[mode]
        lines.append(f"- {NAMES[mode]}：选轮关系 MSE {r['best_fit_relation_mse']:.4f}，末轮 {r['last_fit_relation_mse']:.4f}；"
            f"选轮关系评分平均 AUC {r['best_fit_auc']:.4f}，实际边同类比例 {100*r['best_fit_homophily']:.2f}%。")
    lines+=['','## 邻居与类别表现','']
    for mode in MODES:
        r=a[mode]
        metrics=[v for v in result['class_metrics'] if v['mode']==mode and v['checkpoint']=='best']
        recalls=np.mean([v['recalls'] for v in metrics],axis=0)*100
        lines.append(f"- {NAMES[mode]}：选轮新邻居比例 {100*r['best_new_choices']:.2f}%，末轮 {100*r['last_new_choices']:.2f}%，选轮平均度 {r['best_mean_degree']:.2f}；"
            f"AD/CN/SMCI 召回率分别 {recalls[0]:.2f}% / {recalls[1]:.2f}% / {recalls[2]:.2f}%。")
    lines+=['','预算 10 指对称化前的有向选择数；反向边并集后度数可不同。新邻居比例按全队列及六模态平均，'
        '并非只统计训练或测试患者。','',
        '## 固定参数换回原图','',
        '仅把已训练模型的邻居切回原图，其余参数不变。此项反映模型依赖，不能替代重新训练对照。','']
    for mode in MODES:
        for which,label in (('best','选轮'),('last','末轮')):
            r=[v for v in result['swaps'] if v['mode']==mode and v['checkpoint']==which]
            lines.append(f"- {NAMES[mode]}，{label}：三种子均值 {100*np.mean([v['learned_acc'] for v in r]):.2f}% → "
                f"{100*np.mean([v['original_support_acc'] for v in r]):.2f}%。")
    lines+=['','## 检查与解释范围','',
        '新增 5 项测试通过，覆盖训练标签隔离、自配对排除、类别对平衡、独立构图梯度、共同参数与 CPU/CUDA 随机状态、'
        '固定图下不改变 C 内容传播和检查点重放。已重新加载 18 个新旧检查点，核对全部预测、指标、患者顺序、邻居、源码和数据哈希，'
        '并校验关系损失系数、完整训练轮数和学习率一致。','',
        '本轮是同一批 60 位测试患者上的三个训练种子，沿用历史的第一次最高测试 ACC 选轮，所以只能用于探索；'
        '不是十折结果。只检验了一种关系目标、权重 0.1 和固定预算 10，不能据此否定全部监督构图。'
        '同类关系拟合良好也不能直接证明该邻居对个体预测有用。']
    (OUT/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
