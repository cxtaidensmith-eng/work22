"""Fixed-checkpoint readout interventions on current C R_L115; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from abide_fine_tune import configure, make_model
from relation_experiment import ROOT, setup, dump, sha


def main():
    torch.set_num_threads(1)
    out = ROOT / 'Result/c_current_readout_20260928'
    out.mkdir(exist_ok=False)
    reference = ROOT / 'Result/abide_fine_tune_20260927'
    specification = json.loads((reference / 'best_configuration.json').read_text(encoding='utf-8'))
    for name, digest in specification['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    cfg, dd, data, graphs, _ = setup(-1, 'Config/c_abide_repair_v2_W.ini', 10)
    cfg = configure(cfg, specification['overrides_to_W_config'])
    model = make_model(dd, cfg, graphs, 'content', training_seed=0).eval()
    assert sum(p.numel() for p in model.parameters()) == 726551
    records = []
    for fold in range(1, 11):
        folder = reference / 'runs/R_L115' / f'fold_{fold:02d}'
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        assert (checkpoint['arm'], checkpoint['fold'], checkpoint['seed']) == ('R_L115', fold, 0)
        model.load_state_dict(checkpoint['state_dict'])
        torch.testing.assert_close(model._modal_noise_std * model.noise_scale,
                                   torch.tensor([.15, .15, .15, .03]))
        fit, test = data['Mask'][fold - 1]
        labels = data['Label']
        with torch.no_grad():
            logits = model(data['Feature'])[0]
            with np.load(folder / 'best_predictions.npz') as saved:
                np.testing.assert_array_equal(logits[test].argmax(1).numpy(), saved['logits'].argmax(1))
                np.testing.assert_array_equal(labels[test].numpy(), saved['labels'])
                np.testing.assert_array_equal(np.array(dd['Index'])[test.numpy()], saved['row_index'])
                max_logit_error = float(np.max(np.abs(logits[test].numpy() - saved['logits'])))
            predictions = {'main': logits.argmax(1)}
            for name, module in [('own_only', model.Message_MLP), ('semantic_only', model.Global_Message)]:
                handle = module.register_forward_hook(lambda mod, args, result: torch.zeros_like(result))
                try:
                    predictions[name] = model(data['Feature'])[0].argmax(1)
                finally:
                    handle.remove()
            handle = model.propagation.register_forward_hook(lambda mod, args, result: mod.output_norm(args[0]))
            try:
                predictions['no_graph_residual'] = model(data['Feature'])[0].argmax(1)
            finally:
                handle.remove()
            row = dict(fold=fold, epoch=checkpoint['best']['epoch'],
                       checkpoint=str(folder / 'best.pt'), checkpoint_sha256=sha(folder / 'best.pt'),
                       replay_argmax_exact=True, cpu_gpu_max_logit_difference=max_logit_error, metrics={})
            for name, pred in predictions.items():
                row['metrics'][name] = {
                    split: int((pred[mask] == labels[mask]).sum()) / int(mask.sum())
                    for split, mask in [('train', fit), ('test', test)]}
                row['metrics'][name].update(
                    test_predicted_class_counts=torch.bincount(pred[test], minlength=2).tolist(),
                    corrected_vs_main=int(((pred[test] == labels[test]) & (predictions['main'][test] != labels[test])).sum()),
                    harmed_vs_main=int(((pred[test] != labels[test]) & (predictions['main'][test] == labels[test])).sum()))
            records.append(row)
    summary = {}
    for name in records[0]['metrics']:
        summary[name] = {split: float(np.mean([r['metrics'][name][split] for r in records])) for split in ['train', 'test']}
        for key in ['corrected_vs_main', 'harmed_vs_main']:
            summary[name][key] = sum(r['metrics'][name][key] for r in records)
        summary[name]['test_predicted_class_counts'] = np.sum([r['metrics'][name]['test_predicted_class_counts'] for r in records], axis=0).tolist()
    scope = ('Current R_L115 first maximum-test-ACC checkpoints, seed0 tenfold. CPU threads1, no training, '
             'no threshold or parameter search. All ten original saved test argmax outputs replayed exactly. '
             'Fixed-checkpoint interventions change the learned input distribution and are not trained ablations. '
             'The no_graph_residual intervention preserves propagation.output_norm.')
    result = dict(scope=scope, config=specification, parameters=726551, summary=summary, records=records,
                  script_sha256=sha(__file__))
    dump(out / 'audit.json', result)
    lines = ['# 当前 C / R_L115 固定检查点诊断', '',
             '原十折、seed0、各折最高测试ACC检查点。CPU只读推理，未训练或调参；十折均精确复现已保存测试预测类别。', '',
             '| 推理条件 | 十折ACC | 相对完整模型纠正人数 | 相对完整模型损害人数 | ADS/CN预测计数 |',
             '|---|---:|---:|---:|---|']
    names = dict(main='完整模型', own_only='自身分支（语义输出置零）', semantic_only='语义分支（自身输出置零）', no_graph_residual='关闭图残差、保留LayerNorm')
    for name, row in summary.items():
        lines.append(f"| {names[name]} | {row['test']*100:.4f}% | {row['corrected_vs_main']} | {row['harmed_vs_main']} | {row['test_predicted_class_counts']} |")
    lines += ['', '这些是固定已训练权重下的推理干预，不能替代从头训练的消融，也不能据此认定删除分支后训练会获得相同成绩。', '',
              '完整检查点路径、SHA256、逐折结果与CPU/GPU最大logit误差见 audit.json。']
    (out / '诊断结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
