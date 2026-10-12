"""Complete the fine-search report from saved predictions; no training."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Result/abide_fine_tune_20260927'


def main():
    result=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
    done=json.loads((OUT/'COMPLETE.json').read_text(encoding='utf-8'))
    name=result['retained'];rows=[r for r in result['rows'] if r['arm']==name]
    assert len(rows)==10 and done['all_audits_passed']
    all_rows=[];all_y=[];all_pred=[];all_prob=[]
    for f in range(1,11):
        folder=OUT/'runs'/name/f'fold_{f:02d}'
        with np.load(folder/'best_predictions.npz') as p:
            z=p['logits'];ex=np.exp(z-z.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
            all_rows.extend(p['row_index']);all_y.extend(p['labels']);all_pred.extend(z.argmax(1));all_prob.extend(prob[:,1])
    with np.load(OUT/'data_splits_graphs.npz') as p:
        assert len(all_rows)==len(set(all_rows))==len(p['source_row'])
        np.testing.assert_array_equal(sorted(all_rows),sorted(p['source_row']))
    np.savez_compressed(OUT/'best_oof_predictions.npz',row_index=all_rows,labels=all_y,predictions=all_pred,positive_probability=all_prob)
    summary=dict(name=name,samples=len(all_rows),confusion_matrix=confusion_matrix(all_y,all_pred).tolist(),
        wins=sum(r['acc']>r['baseline_acc'] for r in rows),ties=sum(r['acc']==r['baseline_acc'] for r in rows),
        losses=sum(r['acc']<r['baseline_acc'] for r in rows),corrected=sum(r['corrected'] for r in rows),harmed=sum(r['harmed'] for r in rows),
        new_runs=done['new_runs'],verified_checkpoints=2*done['new_runs'])
    (OUT/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plan=json.loads((OUT/'PLAN.json').read_text());settings=json.loads((OUT/'all_settings.json').read_text())[name]
    config=dict(name=name,overrides_to_W_config=settings,Correlation_effective_noise_sd=.03,
        other_groups_effective_noise_sd=.15,graph_weight_decay=.0005,seed=0,epochs=1000,
        factory='abide_fine_tune.make_model',base_config='Config/c_abide_repair_v2_W.ini',
        source_sha256=plan['code_sha256'],note='Factory applies N1 noise buffer; base config alone does not reproduce this experiment.')
    (OUT/'best_configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['','## 最终核查与逐折结果','',
        f"完成{done['new_runs']}次训练、{2*done['new_runs']}个最佳/末轮检查点复核，最终十折覆盖871名不同数据行且各测试一次。",'',
        '| 折 | 最佳轮次 | N1最佳ACC | 精调最佳ACC | 精调第1000轮ACC |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['fold']} | {r['epoch']} | {100*r['baseline_acc']:.2f}% | {100*r['acc']:.2f}% | {100*r['last_acc']:.2f}% |")
    lines+=['',f"相对N1：{summary['wins']}折提高、{summary['ties']}折持平、{summary['losses']}折降低；纠正{summary['corrected']}例，新增{summary['harmed']}例错误，净多正确{summary['corrected']-summary['harmed']}例。",
        '搜索和晋级使用的第1、9、3、7折平均提高1.7241个百分点；其余6折平均只提高0.1916个百分点，净多正确1例。因此十折收益中较大部分来自参与筛选的折，不能作为独立泛化验证。',
        '最终最佳ACC为89.4383%（折间总体标准差3.0720个百分点），较N1提高0.8046个百分点；末轮提高2.7612个百分点。未达到90%，也不能证明全局最高性能。',
        '本轮保留该配置及所有搜索结果，不覆盖N1和核心C模型。最优已完成配置仅指当前固定seed0、原测试选轮协议与已完成比较范围；需要时可读取best_configuration.json复现。']
    p=OUT/'结果.md';old=p.read_text(encoding='utf-8').split('\n## 最终核查与逐折结果')[0]
    p.write_text(old+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
