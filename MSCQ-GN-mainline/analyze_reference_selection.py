"""Read-only checkpoint diagnostics of the completed reference-selection run.

Writes analysis artifacts only. No retraining, selection, or tuning. Fixed
checkpoint interventions describe dependence, not from-training ablations.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relation_experiment import ROOT, setup, dump, sha, timestamp
from Model.reference_selection import ReferenceSelectionModel
from reference_selection_experiment import folder_for, read


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(count=int(v.size), mean=float(v.mean()), std=float(v.std()),
                quantiles=dict(zip(['min', 'p50', 'p90', 'p95', 'p99', 'max'],
                                  np.quantile(v, [0, .5, .9, .95, .99, 1]).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='Result/reference_selection')
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    protocol = read(out/'PROTOCOL.json')
    if not (out/'COMPLETE.json').exists() or not read(out/'audit.json')['passed']:
        raise ValueError('Require completed, audited experiment')
    for source, digest in protocol['code_sha256'].items():
        if sha(ROOT/source) != digest:
            raise ValueError(f'Experiment source changed: {source}')
    summary = read(out/'summary.json')
    cfg, dd, data, graphs, _ = setup(k=protocol['candidate_k'])
    all_gains, ranges, all_on_off = [], [], []
    targets_verified = 0
    for seed in protocol['seeds']:
        for fold in protocol['folds']:
            folder = folder_for(out, seed, 'crossfit', fold)
            record = read(folder/'complete.json')
            assert sha(folder/'targets.npz') == record['target_sha256']
            with np.load(folder/'targets.npz') as saved:
                assert np.isnan(saved['gains'][saved['test_mask']]).all()
                all_gains.append(saved['gains'][saved['fit_mask']])
            for inner in range(2):
                sub = folder/f'inner_{inner}'
                assert sha(sub/'coalitions.npz') == read(sub/'complete.json')['artifact_sha256']
                with np.load(sub/'coalitions.npz') as saved:
                    losses = saved['losses']
                    ranges.append(losses.max(0)-losses.min(0))
                    all_on_off.append(losses[0]-losses[-1])
            targets_verified += 1
    gain = np.concatenate(all_gains)
    target = 1/(1+np.exp(-np.clip(gain/protocol['gain_temperature'], -80, 80)))
    sensitivity = np.concatenate(ranges)
    target_summary = dict(verified_target_files=targets_verified, modalities=dd['Modal_Name'],
        abs_gain=distribution(np.abs(gain)), abs_gain_by_modality=[distribution(np.abs(gain[:, m])) for m in range(gain.shape[1])],
        target_within_001_of_half=float((np.abs(target-.5) < .01).mean()),
        target_within_005_of_half=float((np.abs(target-.5) < .05).mean()),
        teacher_coalition_nll_range=distribution(sensitivity),
        teacher_nll_range_below_001=float((sensitivity < .001).mean()),
        teacher_nll_range_below_01=float((sensitivity < .01).mean()),
        teacher_all_on_minus_off_gain=distribution(np.concatenate(all_on_off)),
        counting='Patient-modality and patient evaluations repeated across overlapping train folds and seeds')
    modes = ('c_real', 'gate', 'gain_gate')
    gate_values = {mode: [] for mode in modes[1:]}
    by_fold_gate_std = {mode: [] for mode in modes[1:]}
    interventions, metrics = [], {}
    for mode in ('c_real', 'c_random', 'gate', 'gain_gate'):
        rows = [read(folder_for(out, seed, mode, fold)/'complete.json')
                for seed in protocol['seeds'] for fold in protocol['folds']]
        metrics[mode] = dict(best_epoch=distribution([r['best']['epoch'] for r in rows]),
            mean_auc=float(np.mean([r['best']['auc'] for r in rows])),
            mean_macro_f1=float(np.mean([r['best']['macro_f1'] for r in rows])),
            mean_per_class_recall=np.mean([r['best']['per_class_recall'] for r in rows], 0).tolist(),
            parameters=rows[0]['parameters'])
    with torch.no_grad():
        for mode in modes:
            for seed in protocol['seeds']:
                for fold in protocol['folds']:
                    folder = folder_for(out, seed, mode, fold)
                    model = ReferenceSelectionModel(dd, cfg, graphs, seed, learned=mode != 'c_real').to(cfg.Device).eval()
                    checkpoint = (Path(read(folder/'baseline_reference.json')['source'])/'best.pt'
                                  if mode == 'c_real' else folder/'best.pt')
                    model.load_state_dict(torch.load(checkpoint, map_location=cfg.Device, weights_only=False)['state_dict'])
                    logits = model(data['Feature'])[0]
                    test = data['Mask'][fold-1][1]
                    y = data['Label'][test]
                    original = logits[test].argmax(-1)
                    learned = model.propagation.last_gate.clone()
                    with np.load(folder/'predictions.npz') as saved:
                        np.testing.assert_allclose(logits[test].cpu().numpy(), saved['logits'], atol=2e-5, rtol=2e-5)
                        np.testing.assert_array_equal(y.cpu().numpy(), saved['labels'])
                        if mode != 'c_real':
                            np.testing.assert_allclose(learned[test].cpu().numpy(), saved['gates'], atol=2e-5, rtol=2e-5)
                            gate_values[mode].append(saved['gates'])
                            by_fold_gate_std[mode].append(saved['gates'].std(0))
                    correct = original == y
                    nll = float(F.cross_entropy(logits[test], y))
                    masks = dict(all_closed=torch.zeros_like(learned), constant_half=torch.full_like(learned, .5),
                                 all_open=torch.ones_like(learned))
                    if mode != 'c_real':
                        masks['modality_mean'] = learned.mean(0, keepdim=True).expand_as(learned)
                        rng = np.random.default_rng(9001+seed*100+fold)
                        permutation = torch.as_tensor(rng.permutation(len(learned)), device=cfg.Device)
                        masks['patient_permutation'] = learned[permutation]
                    for name, mask in masks.items():
                        predicted = model(data['Feature'], mask)[0][test]
                        updated = predicted.argmax(-1)
                        changed_correct = updated == y
                        interventions.append(dict(mode=mode, seed=seed, fold=fold, intervention=name,
                            base_acc=float(correct.float().mean()), new_acc=float(changed_correct.float().mean()),
                            delta_acc=float(changed_correct.double().mean()-correct.double().mean()),
                            base_nll=nll, new_nll=float(F.cross_entropy(predicted, y)),
                            changed_predictions=int((original != updated).sum()),
                            corrected=int((~correct & changed_correct).sum()), harmed=int((correct & ~changed_correct).sum()),
                            mean_abs_logit_change=float((predicted-logits[test]).abs().mean())))
                    del model
                print(json.dumps(dict(event='checkpoint_diagnostics', mode=mode, seed=seed)), flush=True)
    interventions_summary = []
    for mode in modes:
        names = sorted({r['intervention'] for r in interventions if r['mode'] == mode})
        for name in names:
            rows = [r for r in interventions if r['mode'] == mode and r['intervention'] == name]
            interventions_summary.append(dict(mode=mode, intervention=name, paired_fits=len(rows),
                mean_acc_delta=float(np.mean([r['delta_acc'] for r in rows])),
                mean_nll_delta=float(np.mean([r['new_nll']-r['base_nll'] for r in rows])),
                changed_predictions=sum(r['changed_predictions'] for r in rows),
                corrected=sum(r['corrected'] for r in rows), harmed=sum(r['harmed'] for r in rows),
                counting='598 patients evaluated at three seeds, not 1794 independent patients'))
    gates = {}
    for mode, parts in gate_values.items():
        values = np.concatenate(parts)
        gates[mode] = dict(distribution=distribution(values), mean_by_modality=values.mean(0).tolist(),
            within_001_half=float((np.abs(values-.5) < .01).mean()),
            within_005_half=float((np.abs(values-.5) < .05).mean()),
            exact_zero_fraction=float((values == 0).mean()), exact_one_fraction=float((values == 1).mean()),
            mean_within_fold_patient_std_by_modality=np.mean(by_fold_gate_std[mode], 0).tolist())
    result = dict(analyzed_at=timestamp(), summary=summary, additional_metrics=metrics, target_diagnostics=target_summary,
        learned_gates=gates, checkpoint_interventions=interventions_summary, individual_interventions=interventions,
        source_protocol_sha256=sha(out/'PROTOCOL.json'), analysis_source_sha256=sha(Path(__file__)),
        caveat='Historical test-selected checkpoints; interventions are dependency analysis without retraining. No causal/independent significance claim.')
    dump(out/'analysis.json', result)
    print(json.dumps(dict(target_diagnostics=target_summary, gates=gates, interventions=interventions_summary)), flush=True)


if __name__ == '__main__':
    main()
