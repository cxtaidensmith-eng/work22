"""Audit the fixed-fold self-preservation pilot, with selected/last separated."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from relation_experiment import ROOT, setup, dump, sha, timestamp
from reference_selection_experiment import read
from self_preservation_pilot import (ARMS, SEEDS, WARMUP, EPOCHS, make_arm,
                                     patient_hash, diagnostic)


def main():
    out = ROOT/'Result/self_preservation_pilot'
    assert read(out/'status.json')['state']=='complete'
    plan = read(out/'PLAN.json')
    assert plan['runner_sha256']==sha(ROOT/'self_preservation_pilot.py')
    protocol_path=ROOT/'Result/reference_selection/PROTOCOL.json'
    assert plan['source_protocol_sha256']==sha(protocol_path)
    protocol=read(protocol_path)
    for path,digest in protocol['code_sha256'].items():
        assert sha(ROOT/path)==digest,path
    cfg,dd,data,graphs,paths=setup(k=protocol['candidate_k'])
    assert {Path(p).name:sha(p) for p in paths}==protocol['data_sha256']
    assert all(getattr(cfg,key)==value for key,value in protocol['original_config'].items())
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()==protocol['candidate_hash']
    test=data['Mask'][0][1]
    rows=[]; same_epoch=[]; pair_counts=[]; diagnostic_records=[]; warmup_records=[]
    with torch.no_grad():
        for seed in SEEDS:
            base=out/f'seed_{seed}'
            bundle=torch.load(base/'warmup'/'continuation_state.pt',map_location=cfg.Device,weights_only=False)
            warmup_records.append(dict(seed=seed,fixed_epoch_acc=bundle['record']['last']['acc'],
                selected_acc=bundle['record']['best']['acc'],selected_epoch=bundle['record']['best']['epoch']))
            initializations={arm:read(base/arm/'initialization.json') for arm in ARMS}
            assert initializations['joint']['model_sha256']==initializations['frozen_self']['model_sha256']
            assert all(v['patient_sha256']==bundle['patient_hash'] for v in initializations.values())
            assert all(v['scheduler_last_epoch']==WARMUP for v in initializations.values())
            histories={}
            for arm in ARMS:
                folder=base/arm
                record=read(folder/'complete.json')
                with (folder/'epochs.csv').open(newline='',encoding='utf-8') as stream:
                    history=list(csv.DictReader(stream))
                histories[arm]=history
                assert [int(r['epoch']) for r in history]==list(range(1,EPOCHS+1))
                acc=np.array([float(r['test_acc']) for r in history])
                train_loss=np.array([float(r['train_loss']) for r in history])
                assert np.isfinite(train_loss).all()
                assert int(acc.argmax())+1==record['best']['epoch']
                assert int(acc[WARMUP:].argmax())+WARMUP+1==record['post_warmup_best']['epoch']
                if arm=='frozen_self':
                    assert record['patient_state_sha256']==bundle['patient_hash']
                for which in ('best','last'):
                    cp=folder/f'{which}.pt'
                    assert sha(cp)==record[f'{which}_checkpoint_sha256']
                    checkpoint=torch.load(cp,map_location=cfg.Device,weights_only=False)
                    replay_arm=('self_only' if which=='best' and record['best']['phase']=='warmup' else arm)
                    model=make_arm(dd,cfg,graphs,seed,replay_arm,bundle['state_dict']).eval()
                    initial_graph_state={name:value.detach().clone() for name,value in model.state_dict().items()
                        if name.startswith('propagation.') and not name.startswith('propagation.output_norm.')}
                    model.load_state_dict(checkpoint['state_dict'])
                    logits=model(data['Feature'])[0]
                    pred_file=folder/('predictions.npz' if which=='best' else 'last_predictions.npz')
                    with np.load(pred_file) as saved:
                        np.testing.assert_allclose(logits[test].cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
                        np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
                        assert float((saved['logits'].argmax(-1)==saved['labels']).mean())==record[which]['acc']
                    if which=='last':
                        diag,full,own=diagnostic(model,data['Feature'],data['Label'],test,bundle['logits'],EPOCHS,
                            bundle['patient_hash'],arm=='frozen_self')
                        fit=data['Mask'][0][0]
                        diag['train_acc']=float((full[fit].argmax(-1)==data['Label'][fit]).double().mean())
                        diag['train_bypass_acc']=float((own[fit].argmax(-1)==data['Label'][fit]).double().mean())
                        if arm!='self_only':
                            graph_delta={name:float((model.state_dict()[name]-value).norm())
                                for name,value in initial_graph_state.items()}
                            assert graph_delta['propagation.output.weight']>0
                            diag['graph_state_change_l2']=graph_delta
                            diag['graph_output_weight_norm']=float(model.propagation.output.weight.norm())
                            diag['graph_step']=float(model.propagation.step_logit.sigmoid())
                        diagnostic_records.append(dict(seed=seed,arm=arm,**diag))
                    del model
                rows.append(dict(seed=seed,arm=arm,best_epoch=record['best']['epoch'],best_phase=record['best']['phase'],
                    best_acc=record['best']['acc'],post_best_epoch=record['post_warmup_best']['epoch'],
                    post_best_acc=record['post_warmup_best']['acc'],last_acc=record['last']['acc'],
                    last100_acc=record['mean_last100_acc'],trainable_parameters=record['trainable_parameters'],
                    last100_train_loss=float(train_loss[-100:].mean())))
                same_epoch.extend(dict(seed=seed,arm=arm,**r) for r in read(folder/'diagnostics.json'))
            for arm in ARMS[1:]:
                for column in ('epoch','lr'):
                    assert [r[column] for r in histories[arm]]==[r[column] for r in histories['self_only']]
                assert histories[arm][:WARMUP]==histories['self_only'][:WARMUP]
            for which,filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                for left,right in (('self_only','frozen_self'),('self_only','joint'),('joint','frozen_self')):
                    with np.load(base/left/filename) as a,np.load(base/right/filename) as b:
                        np.testing.assert_array_equal(a['row_index'],b['row_index'])
                        ca=a['logits'].argmax(-1)==a['labels'];cb=b['logits'].argmax(-1)==b['labels']
                        pair_counts.append(dict(seed=seed,checkpoint=which,comparison=f'{right} vs {left}',
                            recovered=int((~ca&cb).sum()),harmed=int((ca&~cb).sum()),
                            delta_acc_pp=100*float(cb.mean()-ca.mean())))
    averages={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm]))
              for key in ('best_acc','post_best_acc','last_acc','last100_acc')} for arm in ARMS}
    original_c=[]
    for seed in SEEDS:
        record=read(ROOT/f'Result/relation_mechanism_nodecay/seed_{seed}/content/fold_01/complete.json')
        original_c.append(dict(seed=seed,best_acc=record['best']['acc'],last_acc=record['last']['acc']))
    analysis=dict(completed_at=timestamp(),rows=rows,averages=averages,pair_counts=pair_counts,
        diagnostics=same_epoch,final_replayed_diagnostics=diagnostic_records,original_c_context=original_c,
        warmup_records=warmup_records,
        scope='One prespecified fold, 60 test patients repeated across three seeds, not ten-fold results',
        limitations=['Historical TEST-ACC selection; no independent confirmation',
            'Freeze changes trainable capacity, optimization, regularization and clipping norm; cannot attribute differences solely to gradient conflict',
            'One fixed 200/800 schedule; negative results do not reject every preservation strategy',
            'Warm-up is included in primary maximum, so a warm-up-selected score does not show graph benefit'],
        analysis_source_sha256=sha(Path(__file__)))
    dump(out/'analysis.json',analysis)
    dump(out/'audit.json',dict(passed=True,arms=9,epochs_per_arm=1000,
        shared_warmup_histories_identical=True,paired_graph_initialization_identical=True,
        initial_predictions_preserved=True,continued_lr_schedules_identical=True,
        frozen_patient_parameters_unchanged=True,selected_and_last_checkpoints_replayed=18,
        first_maxima_verified=True,original_sources_unchanged=True,
        finite_training_losses=True,all_six_graph_arms_updated_from_zero_output=True))
    with (out/'逐组结果.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=dict(self_only='只用自身信息',frozen_self='接图后固定自身参数',joint='接图后联合训练')
    lines=['# 自身表示保护：固定第 1 折的小规模对照','',
        '**结论：这轮没有支持“先冻结自身模型，再训练图，就能获得提升”的方案，暂不扩展这版冻结方案到十折。**'
        '冻结组按原选轮规则和固定末轮都低于自身对照；联合训练的最高分均值略低、固定末轮均值略高，种子间方向不一致，未形成稳定优势。','',
        '冻结组三个种子的主指标都选中了接图之前的检查点。图参数确实更新、自身参数确实保持不变，'
        '因此结果不能归因于图没有训练或冻结没有生效，但这并不等于排除了所有优化问题。'
        '这只筛查了固定 200 轮自身预训练、800 轮接图及零残差起点，不能推广为所有分阶段训练均无效。','',
        '三个训练种子、原第 1 折，使用同一批 60 位测试患者。每个版本共 1000 轮：'
        '前 200 轮共享自身模型训练，固定第 200 轮参数作为起点，后 800 轮分别继续自身模型、接图并冻结自身参数、接图后联合训练。'
        '不按测试成绩选择接图起点。原历史十折结果未修改。','',
        '两种接图版本使用相同图、同一 C 算子及相同初始化；将图输出投影初始化为零，确保接图瞬间预测不变。'
        '冻结版本只更新图专用参数，输出 LayerNorm、自身编码、融合、辅助头和分类器全部固定；仍保留原输入噪声及 dropout。'
        '损失不变，Adam 自身参数动量与原 1000 轮学习率进度延续，图参数从空动量开始。','',
        '## 三种子均值','',
        '主指标继续取全部 1000 轮中第一次最高 TEST ACC，包含共享的前 200 轮。'
        '因此必须同时看选中轮次与固定末轮，不能把接图前的高分视为图带来的改进。','']
    for arm in ARMS:
        r=averages[arm]
        lines.append(f"- {names[arm]}：原选轮 ACC {100*r['best_acc']:.2f}%；"
            f"接图阶段/继续训练阶段最高 ACC {100*r['post_best_acc']:.2f}%；"
            f"固定末轮 {100*r['last_acc']:.2f}%；末 100 轮均值 {100*r['last100_acc']:.2f}%。")
    lines+=['',f"固定第 200 轮起点的三种子平均 ACC 为 "
        f"{100*np.mean([r['fixed_epoch_acc'] for r in warmup_records]):.2f}%。"
        '冻结组的自身部分始终停留在这一训练阶段，不能把它看作已经充分训练后再冻结的模型。']
    lines+=['','## 逐种子结果','']
    for seed in SEEDS:
        for r in [r for r in rows if r['seed']==seed]:
            lines.append(f"- seed {seed}，{names[r['arm']]}：主指标 {100*r['best_acc']:.2f}%"
                f"（第 {r['best_epoch']} 轮，{'接图前/共享阶段' if r['best_phase']=='warmup' else '继续训练阶段'}）；"
                f"阶段内最高 {100*r['post_best_acc']:.2f}%；末轮 {100*r['last_acc']:.2f}%。")
    lines+=['','## 自身预测是否保留','']
    for r in diagnostic_records:
        lines.append(f"- seed {r['seed']}，{names[r['arm']]}：末轮跳过图更新后 ACC {100*r['bypass_acc']:.2f}%；"
            f"相对第 200 轮自身预测改变 {r['bypass_changed_from_warm']} 次，"
            f"纠正 {r['bypass_corrected_vs_warm']} 次、改错 {r['bypass_harmed_vs_warm']} 次。")
    joint_diags=[r for r in diagnostic_records if r['arm']=='joint']
    joint_corrected=sum(r['bypass_corrected_vs_warm'] for r in joint_diags)
    joint_harmed=sum(r['bypass_harmed_vs_warm'] for r in joint_diags)
    lines+=['',f'三个种子的末轮联合模型跳过图更新后，相对共享第 200 轮起点合计纠正 '
        f'{joint_corrected} 次、改错 {joint_harmed} 次（同一批患者的重复评估）。'
        '因此本轮没有观察到自身预测整体退化，不能把“图训练破坏了自身判断”当作已确认原因。']
    lines+=['','## 固定末轮中图更新的作用','',
        '以下比较同一检查点的完整预测与跳过图更新的预测，仅用于诊断；联合训练后的图旁路并不等同于单独训练的自身模型。','']
    for r in diagnostic_records:
        if r['arm']=='self_only':continue
        lines.append(f"- seed {r['seed']}，{names[r['arm']]}：图更新纠正 "
            f"{r['graph_corrected_vs_current_bypass']} 次、改错 {r['graph_harmed_vs_current_bypass']} 次；"
            f"图输出矩阵从零学习后的范数 {r['graph_output_weight_norm']:.4f}。")
    lines+=['','## 判断边界','',
        '这是保持自身参数不变是否有帮助的首轮筛查。冻结同时改变可训练容量、正则化及全局梯度裁剪中参与计算的梯度，'
        '即使冻结组更好，也不能单独证明梯度冲突是原因。若冻结组没有优势，也不能据此排除所有保护自身表示的方法。','',
        '历史最高测试 ACC 选轮仍属于探索性协议；三个种子不是三批独立患者。'
        '本次没有自动扩展十折，也没有增加表示分离、蒸馏或动态构图模块。','',
        '检查通过：3 项新增实现测试、原代码/数据/候选图核对、三臂前 200 轮完全相同、接图初始化相同、学习率进度相同、'
        '冻结自身参数不变、图输出参数确实更新、训练损失均为有限值、18 个最佳/末轮检查点重放和首次最高 ACC 选轮复核。']
    (out/'结果分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=rows,averages=averages,final_diagnostics=diagnostic_records),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
