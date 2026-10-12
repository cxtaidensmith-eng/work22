"""Read-only checkpoint/record audit of existing C runs; never trains a model.

Writes only a fresh audit directory, leaving all experiment artifacts untouched.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score

from relation_experiment import ROOT, setup
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Utils import SET_Random
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


BASE = ROOT/'Result/relation_mechanism_nodecay'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_metrics(z, y):
    predicted = z.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    return dict(acc=int((predicted == y).sum())/len(y),
        f1=float(f1_score(y, predicted, average='weighted', zero_division=0)),
        macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)),
        auc=float(roc_auc_score(np.eye(3)[y], z)),
        per_class_recall=recall_score(y, predicted, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        confusion_matrix=matrix.tolist())


def check_metrics(actual, expected):
    for key in ('acc','f1','macro_f1','auc'):
        assert abs(actual[key]-expected[key]) < 1e-12, (key, actual[key], expected[key])
    np.testing.assert_array_equal(actual['confusion_matrix'], expected['confusion_matrix'])
    np.testing.assert_allclose(actual['per_class_recall'], expected['per_class_recall'], atol=1e-12, rtol=0)


def loss_label_probe(outputs, labels, fit, test, dd, cfg):
    """No optimizer or training forward: test whether held-out targets enter CE.

    Keep the historical cohort class weights fixed; their separate use of label
    frequencies is explicitly disclosed in the report.
    """
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    changed = labels.clone()
    changed[test] = (changed[test]+1) % 3
    results = []
    for y in (labels, changed):
        logits = outputs[0].detach().clone().requires_grad_(True)
        embeddings = [z.detach().clone().requires_grad_(True) for z in outputs[1]]
        heads = [z.detach().clone().requires_grad_(True) for z in outputs[2]]
        inputs = (logits, *embeddings, *heads)
        loss = criterion(logits, y, fit, embeddings, heads)
        grads = torch.autograd.grad(loss, inputs)
        results.append((float(loss.detach()), [g.detach() for g in grads]))
    assert results[0][0] == results[1][0]
    assert all(torch.equal(a,b) for a,b in zip(results[0][1], results[1][1]))
    gradients = results[0][1]
    for g in (gradients[0], *gradients[1+len(outputs[1]):]):
        assert int(torch.count_nonzero(g[test])) == 0
    return dict(passed=True, fixed_cohort_class_weights=True,
                test_target_changes_leave_loss_and_all_output_gradients_unchanged=True,
                test_rows_have_zero_final_and_auxiliary_logit_supervised_gradients=True,
                note='All-cohort unlabeled embedding regularizer and transductive feature paths remain active')


def main():
    started = datetime.now(timezone.utc)
    out = ROOT/'Result'/('c_result_reaudit_'+started.strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    checks = []

    def freeze(path):
        path = Path(path)
        manifest[str(path.relative_to(ROOT))] = sha(path)

    cfg, dd, data, graphs, paths = setup()
    protocol = read(BASE/'seed_0/PROTOCOL.json')
    for name,digest in protocol['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
        freeze(ROOT/name)
    for path in paths:
        assert sha(path) == protocol['data_sha256'][Path(path).name]
        freeze(path)
    alternate = read(ROOT/'Result/alternating_cooperative_seed0_20260923/PLAN.json')
    for name,digest in alternate['reference_manifest'].items():
        assert sha(ROOT/name) == digest, name
        freeze(ROOT/name)
    assert cfg.epochs == protocol['epochs'] == 1000
    assert cfg.seed == protocol['split_seed'] == 0
    assert torch.__version__ == protocol['torch'] and torch.cuda.is_available()
    for key,value in protocol['original_config'].items():
        assert getattr(cfg,key) == value, key

    frame = pd.read_csv(paths[0])
    source_rows = np.asarray(dd['Index'])
    labels = data['Label'].cpu().numpy()
    features = data['Feature'].cpu().numpy()
    np.testing.assert_array_equal(features, frame.iloc[source_rows,:-1].to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(labels, frame.iloc[source_rows,-1].to_numpy(dtype=np.int64)-1)
    assert features.shape == (598,360) and np.isfinite(features).all()
    assert sorted(source_rows.tolist()) == list(range(598))
    assert sorted(i for group in dd['Modal_Index'] for i in group) == list(range(360))
    assert frame.columns[-1] == 'LABEL'
    test_masks = np.stack([test.cpu().numpy() for _,test in data['Mask']])
    np.testing.assert_array_equal(test_masks.sum(0), np.ones(598, dtype=int))
    for fit,test in data['Mask']:
        assert not bool((fit & test).any()) and bool((fit | test).all())
    np.testing.assert_allclose(dd['Label_Weight'].cpu().numpy(), (598-np.bincount(labels))/598, atol=1e-7)
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    assert graph_hash == protocol['candidate_hash']
    duplicate_rows = int(frame.iloc[:,:-1].duplicated(keep=False).sum())
    checks.append('Original data, labels, modal indices, full row order, masks and graph support match')

    seed_results, all_rows, probe = [], [], None
    legacy_summary = read(BASE/'summary.json')
    freeze(BASE/'summary.json'); freeze(BASE/'PLAN.json'); freeze(Path(__file__))
    checkpoint_hashes = []
    for seed in (0,1,2):
        old = read(BASE/f'seed_{seed}/PROTOCOL.json')
        freeze(BASE/f'seed_{seed}/PROTOCOL.json')
        assert old['training_seed'] == seed and old['split_seed'] == 0 and old['epochs'] == 1000
        for key in ('original_config','data_sha256','candidate_hash','code_sha256'):
            assert old[key] == protocol[key], key
        with np.load(BASE/f'seed_{seed}/splits_and_graphs.npz') as saved:
            np.testing.assert_array_equal(saved['source_row'], source_rows)
            np.testing.assert_array_equal(saved['labels'], labels)
            np.testing.assert_array_equal(saved['test_masks'], test_masks)
            for i,g in enumerate(graphs):
                np.testing.assert_array_equal(saved[f'edge_{i}'], g['edge'].numpy())
                np.testing.assert_array_equal(saved[f'neighbors_{i}'], g['neighbors'].numpy())
        freeze(BASE/f'seed_{seed}/splits_and_graphs.npz')
        SET_Random(seed)
        model = RelationSheafModel(dd,cfg,graphs,'content',training_seed=seed).to(cfg.Device).eval()
        optimizer_groups = optimizer_manifest(model, relation_optimizer(model,cfg,0.))
        assert sum(p.numel() for p in model.parameters()) == 895096
        assert model.propagation.mode == 'content'
        assert not any(hasattr(model,n) for n in ('GCN','Adj_Learning','fixed_adj'))
        seed_rows, covered, best_predictions, last_predictions = [], [], [], []
        for fold in range(1,11):
            folder = BASE/f'seed_{seed}/content/fold_{fold:02d}'
            record, best_json = read(folder/'complete.json'), read(folder/'best.json')
            assert (record['mode'],record['fold'],record['seed']) == ('content',fold,seed)
            assert record['parameters'] == 895096 and record['best'] == best_json
            assert read(folder/'optimizer_groups.json') == optimizer_groups
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                curve = list(csv.DictReader(stream))
            assert [int(v['epoch']) for v in curve] == list(range(1,1001))
            assert np.isfinite([[float(v[k]) for k in ('train_loss','test_loss','test_acc','lr')] for v in curve]).all()
            acc = np.array([float(v['test_acc']) for v in curve])
            assert int(acc.argmax())+1 == record['best']['epoch']
            assert acc.max() == record['best']['acc'] and acc[-1] == record['last']['acc']
            assert record['last']['epoch'] == 1000
            fit,test = data['Mask'][fold-1]
            mask = test.cpu().numpy()
            n = int(mask.sum())
            np.testing.assert_allclose(acc*n, np.rint(acc*n), atol=1e-10, rtol=0)
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            assert (checkpoint['mode'],checkpoint['fold'],checkpoint['seed']) == ('content',fold,seed)
            assert checkpoint['best'] == best_json
            checkpoint_hashes.append(sha(folder/'best.pt'))
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            saved_stages = {}
            for stage,filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                with np.load(folder/filename) as saved:
                    z,y,rows = saved['logits'],saved['labels'],saved['row_index']
                    assert z.shape == (n,3) and np.isfinite(z).all()
                    np.testing.assert_array_equal(rows, source_rows[mask])
                    np.testing.assert_array_equal(y, labels[mask])
                    actual = independent_metrics(z,y)
                    check_metrics(actual, record[stage])
                    saved_stages[stage] = (z.copy(),y.copy())
            with torch.no_grad():
                gpu_outputs = model(data['Feature'])
            gpu = gpu_outputs[0][test].cpu().numpy()
            expected,y = saved_stages['best']
            np.testing.assert_allclose(gpu,expected,atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(gpu.argmax(-1), expected.argmax(-1))
            check_metrics(independent_metrics(gpu,y), record['best'])
            if probe is None:
                probe = loss_label_probe(gpu_outputs,data['Label'],fit,test,dd,cfg)
            # Independent CPU evaluation of exactly the same frozen weights.
            model.to('cpu')
            with torch.no_grad():
                cpu = model(data['Feature'].cpu())[0][mask].numpy()
            np.testing.assert_allclose(cpu,expected,atol=5e-5,rtol=5e-5)
            np.testing.assert_array_equal(cpu.argmax(-1), expected.argmax(-1))
            check_metrics(independent_metrics(cpu,y), record['best'])
            model.to(cfg.Device)
            item = dict(seed=seed,fold=fold,train_rows=int(fit.sum()),test_rows=n,
                best_epoch=record['best']['epoch'], best_correct=int((expected.argmax(-1)==y).sum()),
                best_acc=record['best']['acc'],last_acc=record['last']['acc'],
                mean_last100_acc=float(acc[-100:].mean()),
                gpu_max_abs_logit_error=float(np.abs(gpu-expected).max()),
                cpu_max_abs_logit_error=float(np.abs(cpu-expected).max()),
                changed_gpu_predictions=0,changed_cpu_predictions=0,
                best_checkpoint_sha256=sha(folder/'best.pt'))
            seed_rows.append(item); all_rows.append(item)
            covered.extend(source_rows[mask].tolist())
            best_predictions.extend((expected.argmax(-1)==y).tolist())
            last_z,last_y = saved_stages['last']
            last_predictions.extend((last_z.argmax(-1)==last_y).tolist())
            for filename in ('best.pt','best.json','complete.json','epochs.csv','predictions.npz',
                             'last_predictions.npz','optimizer_groups.json'):
                freeze(folder/filename)
            print(json.dumps(dict(event='verified',seed=seed,fold=fold,best_correct=item['best_correct'],
                                  gpu_error=item['gpu_max_abs_logit_error'],cpu_error=item['cpu_max_abs_logit_error'])),flush=True)
        assert sorted(covered) == list(range(598))
        summary = dict(seed=seed,folds=10,records=seed_rows,
            mean_best_acc=float(np.mean([v['best_acc'] for v in seed_rows])),
            mean_last_acc=float(np.mean([v['last_acc'] for v in seed_rows])),
            mean_last100_acc=float(np.mean([v['mean_last100_acc'] for v in seed_rows])),
            pooled_best_correct=sum(best_predictions), pooled_best_acc=sum(best_predictions)/598,
            pooled_last_correct=sum(last_predictions), samples=598)
        assert abs(summary['mean_best_acc']-legacy_summary['variants']['content']['seed_means'][str(seed)]['acc']) < 1e-12
        seed_results.append(summary)
        del model,gpu_outputs
        torch.cuda.empty_cache()
    assert len(set(checkpoint_hashes)) == 30
    all_seed_mean = float(np.mean([s['mean_best_acc'] for s in seed_results]))
    assert abs(all_seed_mean-legacy_summary['variants']['content']['mean_acc_complete_seeds']) < 1e-12
    assert abs(seed_results[0]['mean_best_acc']-read(ROOT/'Result/alternating_cooperative_seed0_20260923/summary.json')['variants']['C']['best']['acc']) < 1e-12
    for name,digest in manifest.items():
        assert sha(ROOT/name) == digest, f'Artifact changed during audit: {name}'
    report = dict(passed=True,started_at=started.isoformat(),completed_at=datetime.now(timezone.utc).isoformat(),
        training_runs_started=0,checkpoint_replays=dict(distinct_best_checkpoints=30,gpu=30,cpu=30),
        verified_epoch_rows=30000,verified_prediction_files=60,
        source_and_result_sha256=manifest,checks=checks,seed_results=seed_results,
        mean_best_acc_across_existing_three_seeds=all_seed_mean,
        direct_supervised_loss_probe=probe,
        data_checks=dict(rows=598,features=360,label_counts=np.bincount(labels).tolist(),
            every_row_tested_once_per_seed=True,train_test_disjoint_within_fold=True,
            exact_duplicate_feature_rows=duplicate_rows,label_column_excluded_from_inputs=True),
        limits=['Historical TEST ACC selects epoch, so not an independent held-out estimate',
                'Transductive full-cohort features/graphs and unlabeled embedding regularizer retained',
                'Class weights use full-cohort label frequencies, not train-only frequencies',
                'Processed CSV lacks patient IDs; row separation does not establish longitudinal subject separation',
                'Raw preprocessing provenance/fit scope cannot be established from the processed CSV and current loader',
                'No last.pt exists for these C runs: last metrics checked from saved logits and logs only',
                'Checkpoint inference replay, not a fresh training reproduction'],
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'fold_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    seed0=seed_results[0]
    lines=['# C 结果独立复核','',
        f"结论：seed 0 十折最佳测试 ACC 复算为 {100*seed0['mean_best_acc']:.8f}%，原报告94.98%无计算或检查点对应错误。",'',
        '本次未启动训练。回放了已有三个种子的30个best检查点，每个分别在GPU和CPU上推理；'
        '全部预测类别一致，核查30,000条epoch日志和60份best/last预测文件。','',
        f"GPU最大logit绝对误差 {max(v['gpu_max_abs_logit_error'] for v in all_rows):.9g}；"
        f"CPU最大误差 {max(v['cpu_max_abs_logit_error'] for v in all_rows):.9g}，均未改变分类。",'',
        '| seed 0 折 | 测试样本数 | 正确数 | 首次最佳轮次 | 最佳ACC |',
        '|---|---:|---:|---:|---:|']
    for v in seed0['records']:
        lines.append(f"| {v['fold']} | {v['test_rows']} | {v['best_correct']} | {v['best_epoch']} | {100*v['best_acc']:.4f}% |")
    lines += ['',f"合计判对{seed0['pooled_best_correct']}/598行，合并计数ACC为{100*seed0['pooled_best_acc']:.8f}%；"
        '原指标按折等权平均，因前8折60行、后2折59行，两者略有差别。', '',
        '| 已有训练种子 | 十折最佳ACC | 十折第1000轮ACC |','|---|---:|---:|']
    for v in seed_results:
        lines.append(f"| {v['seed']} | {100*v['mean_best_acc']:.4f}% | {100*v['mean_last_acc']:.4f}% |")
    lines += ['',f"已有三种子最佳ACC均值为{100*all_seed_mean:.4f}%。94.98%仅是seed 0成绩；"
        'A1当前只有seed 0结果，配对比较仍须使用C的seed 0，不能与C的三种子均值混比。','',
        '核查边界：','',
        '- LABEL列未进入模型输入，模态索引覆盖360个特征；每个种子598行均恰好测试一次，同折训练/测试无交集。',
        f'- 完全重复的特征行数：{duplicate_rows}。CSV缺少患者ID，不能仅凭行号排除同一患者不同随访跨折。',
        '- 固定原类别权重时，更改测试标签不会改变训练损失或其输出梯度；最终及辅助分类的测试行监督梯度为零。',
        '- 原协议仍使用测试ACC选轮、全队列标签频率计算类别权重、全队列特征构图和无标签正交正则。不能据此宣称完全没有测试信息参与。',
        '- 标准化/填补等上游预处理的拟合范围，当前处理后CSV与加载代码不足以追溯。',
        '- C没有last.pt；第1000轮仅核对保存预测及日志，没有声称回放末轮模型。',
        '- 本次是保存权重的推理复现，没有重新训练，不保证重新训练每一位数完全相同。','',
        '原始结果与源码均未修改。完整哈希、逐折误差和检查范围见audit.json和fold_audit.csv。','']
    (out/'复核报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(passed=True,output=str(out),seed0_acc=seed0['mean_best_acc'],
        three_seed_mean=all_seed_mean,max_gpu_error=max(v['gpu_max_abs_logit_error'] for v in all_rows),
        max_cpu_error=max(v['cpu_max_abs_logit_error'] for v in all_rows)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
