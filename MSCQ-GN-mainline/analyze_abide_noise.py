"""Read saved, replay-verified predictions; no model fitting or tuning."""
import json
import csv
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score,roc_auc_score,confusion_matrix
from relation_experiment import ROOT,dump,sha
from alternating_cooperative_experiment import read,verify_manifest

OUT=ROOT/'Result/abide_noise_adaptation_20260927'
REF=ROOT/'Result/c_abide_repair_v2_20260927/full/W'


def metrics(folder,fold,rows,labels,masks):
    result=read(folder/'complete.json');audit=read(folder/'noise_and_prediction_audit.json')
    assert result['audit_passed'] and audit['passed']
    with (folder/'epochs.csv').open() as stream:curve=list(csv.DictReader(stream))
    assert [int(r['epoch']) for r in curve]==list(range(1,1001))
    assert int(np.argmax([float(r['test_acc']) for r in curve]))+1==result['best']['epoch']
    with np.load(folder/'best_predictions.npz') as p:
        z=p['logits'];y=p['labels'];rid=p['row_index']
    np.testing.assert_array_equal(rid,rows[masks[fold-1]])
    np.testing.assert_array_equal(y,labels[masks[fold-1]])
    pred=z.argmax(1);ex=np.exp(z-z.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
    assert float((pred==y).mean())==result['best']['acc']
    with np.load(REF/f'content/fold_{fold:02d}/predictions.npz') as p:
        np.testing.assert_array_equal(p['row_index'],rid);np.testing.assert_array_equal(p['labels'],y)
        old=p['logits'].argmax(1)
    return dict(fold=fold,epoch=result['best']['epoch'],acc=float((pred==y).mean()),
        baseline_acc=float((old==y).mean()),macro_f1=float(f1_score(y,pred,average='macro')),
        balanced_acc=float(balanced_accuracy_score(y,pred)),probability_auc=float(roc_auc_score(y,prob[:,1])),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
        corrected=int(((old!=y)&(pred==y)).sum()),harmed=int(((old==y)&(pred!=y)).sum()),
        last_acc=result['last']['test_acc'],last_train_acc=result['last']['train_acc'],
        late100_mean_acc=result['late100_mean_acc'],best_checkpoint_sha256=sha(folder/'best.pt')),(rid,y,pred,prob[:,1])


def main():
    done=read(OUT/'COMPLETE.json');plan=read(OUT/'PLAN.json')
    verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    assert sha(OUT/'data_splits_graphs.npz')==plan['split_graph_archive_sha256']
    with np.load(OUT/'data_splits_graphs.npz') as p:rows=p['source_row'];labels=p['labels'];masks=p['test_masks']
    pilots={}
    for arm in ['N1','N2','N3']:
        rr=[metrics(OUT/'pilot'/arm/f'fold_{f:02d}',f,rows,labels,masks)[0] for f in [1,9]]
        pilots[arm]=dict(folds=rr,mean_acc=float(np.mean([r['acc'] for r in rr])))
    result=dict(pilot=pilots,selection=read(OUT/'selection.json'),full=None,script_sha256=sha(__file__))
    lines=['','## 预测与机制核查','',
        '全部已完成训练均核对1000轮轨迹、首次最佳选轮、噪声缓冲区及最佳/末轮预测。N1—N3只改训练输入噪声，推理时均无新增噪声；网络、参数量、数据和图未改变。']
    for arm,group in pilots.items():
        corrected=sum(r['corrected'] for r in group['folds']);harmed=sum(r['harmed'] for r in group['folds'])
        lines.append(f'- {arm}相对W两折：纠正{corrected}例，新增{harmed}例错误，净变化{corrected-harmed:+d}例。')
    selected=done['selected']
    if selected:
        rr=[];preds=[]
        for f in range(1,11):
            record,pred=metrics(OUT/'full'/selected/f'fold_{f:02d}',f,rows,labels,masks);rr.append(record);preds.append(pred)
        all_rows=np.concatenate([p[0] for p in preds]);all_y=np.concatenate([p[1] for p in preds]);all_p=np.concatenate([p[2] for p in preds]);all_prob=np.concatenate([p[3] for p in preds])
        assert len(all_rows)==len(rows) and len(np.unique(all_rows))==len(rows)
        np.testing.assert_array_equal(np.sort(all_rows),np.sort(rows))
        full=dict(arm=selected,folds=rr,mean_acc=float(np.mean([r['acc'] for r in rr])),std_acc=float(np.std([r['acc'] for r in rr])),
            mean_macro_f1=float(np.mean([r['macro_f1'] for r in rr])),mean_balanced_acc=float(np.mean([r['balanced_acc'] for r in rr])),
            mean_probability_auc=float(np.mean([r['probability_auc'] for r in rr])),mean_last_acc=float(np.mean([r['last_acc'] for r in rr])),
            mean_late100_acc=float(np.mean([r['late100_mean_acc'] for r in rr])),baseline_mean_acc=float(np.mean([r['baseline_acc'] for r in rr])),
            corrected=sum(r['corrected'] for r in rr),harmed=sum(r['harmed'] for r in rr),
            pooled_acc=float((all_p==all_y).mean()),confusion_matrix=confusion_matrix(all_y,all_p,labels=[0,1]).tolist(),
            per_fold_wins=sum(r['acc']>r['baseline_acc'] for r in rr),per_fold_ties=sum(r['acc']==r['baseline_acc'] for r in rr),
            other8_mean_delta=float(np.mean([r['acc']-r['baseline_acc'] for r in rr if r['fold'] not in [1,9]])))
        result['full']=full
        np.savez_compressed(OUT/'selected_oof_predictions.npz',row_index=all_rows,labels=all_y,predictions=all_p,positive_probabilities=all_prob)
        lines+=['','## 完整十折','',f"固定候选{selected}，十折最佳ACC：{100*full['mean_acc']:.4f}% ± {100*full['std_acc']:.4f}%（折间总体标准差）。W为{100*full['baseline_mean_acc']:.4f}%。",
            f"Macro-F1 {100*full['mean_macro_f1']:.4f}%；balanced ACC {100*full['mean_balanced_acc']:.4f}%；概率AUC {100*full['mean_probability_auc']:.4f}%。",
            f"第1000轮均值{100*full['mean_last_acc']:.4f}%，最后100轮均值{100*full['mean_late100_acc']:.4f}%。",
            f"相对W纠正{full['corrected']}例、新增{full['harmed']}例错误；{full['per_fold_wins']}折提高、{full['per_fold_ties']}折持平。另8折平均变化{100*full['other8_mean_delta']:+.4f}个百分点。",'',
            '| 折 | 最佳轮次 | W ACC | 当前ACC | 第1000轮ACC |','|---|---:|---:|---:|---:|']
        for r in rr:lines.append(f"| {r['fold']} | {r['epoch']} | {r['baseline_acc']*100:.2f}% | {r['acc']*100:.2f}% | {r['last_acc']*100:.2f}% |")
    lines+=['','保留原最高测试ACC选轮规则，且候选经过两折筛选。这是探索性优化结果，不是独立验证或全局最优证明。无论结果如何，本轮不追加新的噪声参数。']
    dump(OUT/'analysis.json',result)
    report=OUT/'结果.md';old=report.read_text(encoding='utf-8').split('\n## 预测与机制核查')[0]
    report.write_text(old+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(pilot={a:g['mean_acc'] for a,g in pilots.items()},selected=selected,
        full=None if result['full'] is None else {k:v for k,v in result['full'].items() if k!='folds'}),ensure_ascii=False))


if __name__=='__main__':main()
