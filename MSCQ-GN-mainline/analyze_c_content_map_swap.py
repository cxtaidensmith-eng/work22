"""Frozen Framework C: swap one content's complete send/receive map pair.

All relation sources for the target content receive the donor's four map
parameters. Features, graph support, edge weights and other parameters stay
fixed. The original operator recomputes normalization with the swapped scales.
This is a checkpoint dependency diagnostic, not a from-training ablation.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import Model.relation_sheaf as sheaf_module
from Model.relation_sheaf import RelationSheafModel, conformal_maps
from analyze_c_edge_content_weights import REFERENCE, verify_inputs
from relation_experiment import ROOT, dump, setup, sha, timestamp


DEFAULT_OUTPUT = ROOT / 'Result/c_content_map_swap_20261010'
PREVIOUS = ROOT / 'Result/c_edge_content_weights_20261009'


def model_fingerprint(model):
    digest = hashlib.sha256()
    for category, items in [('parameters', model.named_parameters()), ('buffers', model.named_buffers())]:
        for name, tensor in sorted(items):
            a = tensor.detach().cpu().contiguous().numpy()
            digest.update((category + name + str(a.dtype) + str(a.shape)).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def array_hash(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def classification_values(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    maximum = logits.max(-1)
    logsum = maximum + np.log(np.exp(logits - maximum[..., None]).sum(-1))
    true_logits = np.take_along_axis(logits, labels[..., None], -1)[..., 0]
    nll = logsum - true_logits
    other = logits.copy()
    np.put_along_axis(other, labels[..., None], -np.inf, -1)
    margin = true_logits - other.max(-1)
    return nll, margin, logits.argmax(-1)


@torch.no_grad()
def analyze_fold(fold, cfg, dd, data, graphs, output):
    m, n, hidden = len(graphs), dd['Sample_Num'], cfg.Hidden_size
    folder = REFERENCE / 'content' / f'fold_{fold:02d}'
    checkpoint = torch.load(folder / 'best.pt', map_location=cfg.Device, weights_only=False)
    assert checkpoint['mode'] == 'content' and checkpoint['seed'] == 0 and checkpoint['fold'] == fold
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 895096
    p = model.propagation
    state_before = model_fingerprint(model)
    hook_counts_before = [(len(module._forward_pre_hooks), len(module._forward_hooks))
                          for module in model.modules()]
    captured = {}
    intervention = [None, None]
    original_map_values = p.map_values
    original_operator = sheaf_module.cooperative_operator

    def map_wrapper(z):
        raw = original_map_values(z)
        target, donor = intervention
        if target is not None:
            raw = raw.clone()
            raw[:, target, :, :] = raw[:, donor, :, :]
        captured['maps'] = raw.detach().clone()
        return raw

    def operator_wrapper(x, edge, weight, *args):
        captured['weight'] = weight.detach().clone()
        captured['edge'] = edge.detach()
        return original_operator(x, edge, weight, *args)

    def pre_hook(module, args):
        captured['tokens'] = args[0].detach().clone()

    def post_hook(module, args, result):
        captured['propagation'] = result.detach().clone()

    pre = p.register_forward_pre_hook(pre_hook)
    post = p.register_forward_hook(post_hook)
    p.map_values = map_wrapper
    sheaf_module.cooperative_operator = operator_wrapper
    started = time.perf_counter()
    test = data['Mask'][fold - 1][1]
    test_np = test.cpu().numpy()
    labels = data['Label'][test].cpu().numpy()
    patient_ordinals = np.flatnonzero(test_np)
    count = len(labels)
    try:
        baseline_logits_all = model(data['Feature'])[0].detach()
        baseline_logits = baseline_logits_all[test].cpu().numpy()
        baseline = {key: value.clone() for key, value in captured.items()}
        assert baseline['maps'].shape == (m, m, n, 4)
        torch.testing.assert_close(baseline['maps'],
            baseline['maps'][0, :, 0, :][None, :, None, :].expand(m, m, n, 4), atol=0, rtol=0)
        with np.load(folder / 'predictions.npz', allow_pickle=False) as original:
            np.testing.assert_array_equal(original['labels'], labels)
            np.testing.assert_array_equal(original['row_index'], np.asarray(dd['Index'])[test_np])
            np.testing.assert_allclose(baseline_logits, original['logits'], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(baseline_logits.argmax(1), original['logits'].argmax(1))
            replay_error = float(np.abs(baseline_logits - original['logits']).max())
        with np.load(PREVIOUS / f'fold_{fold:02d}.npz', allow_pickle=False) as prior:
            groups = p.edge_group.cpu().numpy()
            receivers = (p.edge[0] % n).cpu().numpy()
            weights = baseline['weight'].cpu().numpy()
            for r in range(m):
                for c in range(m):
                    selected = (groups == r * m + c) & test_np[receivers]
                    np.testing.assert_array_equal(weights[selected], prior[f'r{r}_weight'][c])
        baseline_propagation = baseline['propagation'][test].cpu().numpy()
        baseline_nll, baseline_margin, baseline_prediction = classification_values(baseline_logits, labels)
        logits = np.empty((m, m, count, baseline_logits.shape[-1]), dtype=np.float32)
        target_outputs = np.empty((m, m, count, hidden), dtype=np.float32)
        relative_l2 = np.empty((m, m, count), dtype=np.float64)
        absolute_l2 = np.empty_like(relative_l2)
        for target in range(m):
            for donor in range(m):
                intervention[:] = [target, donor]
                captured.clear()
                actual_logits_all = model(data['Feature'])[0].detach()
                assert torch.equal(captured['tokens'], baseline['tokens'])
                assert torch.equal(captured['edge'], baseline['edge'])
                assert torch.equal(captured['weight'], baseline['weight'])
                expected_maps = baseline['maps'].clone()
                expected_maps[:, target] = baseline['maps'][:, donor]
                assert torch.equal(captured['maps'], expected_maps)
                keep = [content for content in range(m) if content != target]
                assert torch.equal(captured['propagation'][:, keep], baseline['propagation'][:, keep])
                if target == donor:
                    assert torch.equal(actual_logits_all, baseline_logits_all)
                    assert torch.equal(captured['propagation'], baseline['propagation'])
                logits[target, donor] = actual_logits_all[test].cpu().numpy()
                new = captured['propagation'][test, target].cpu().numpy()
                target_outputs[target, donor] = new
                old = baseline_propagation[:, target]
                difference = np.linalg.norm(new.astype(np.float64) - old.astype(np.float64), axis=-1)
                denominator = np.maximum(np.linalg.norm(old.astype(np.float64), axis=-1), 1e-12)
                absolute_l2[target, donor] = difference
                relative_l2[target, donor] = difference / denominator
            print(json.dumps(dict(fold=fold, completed_targets=target + 1, swaps_completed=(target + 1) * m,
                seconds=time.perf_counter() - started, no_training=True)), flush=True)
        all_labels = np.broadcast_to(labels, (m, m, count))
        nll, margin, prediction = classification_values(logits, all_labels)
        delta_nll = nll - baseline_nll[None, None, :]
        delta_margin = margin - baseline_margin[None, None, :]
        changed = prediction != baseline_prediction[None, None, :]
        baseline_correct = baseline_prediction == labels
        correct = prediction == labels[None, None, :]
        corrected = ~baseline_correct[None, None, :] & correct
        harmed = baseline_correct[None, None, :] & ~correct
        diagonal = np.arange(m)
        for values in (delta_nll, delta_margin, changed, corrected, harmed, relative_l2, absolute_l2):
            assert np.all(values[diagonal, diagonal] == 0)
        raw_maps = baseline['maps'][0, :, 0].cpu().numpy()
        send, receive, send_scale, receive_scale = conformal_maps(torch.from_numpy(raw_maps))
        effective = torch.bmm(receive.transpose(-1, -2), send).numpy()
        arrays = dict(fold=np.asarray(fold), modality_names=np.asarray(dd['Modal_Name'], dtype='U32'),
                      heldout_row_ordinals=patient_ordinals, labels=labels,
                      baseline_logits=baseline_logits, baseline_prediction=baseline_prediction,
                      baseline_nll=baseline_nll, baseline_margin=baseline_margin,
                      baseline_propagation=baseline_propagation,
                      swapped_logits=logits, swapped_prediction=prediction,
                      swapped_nll=nll, swapped_margin=margin, delta_nll=delta_nll, delta_margin=delta_margin,
                      changed=changed, corrected=corrected, harmed=harmed,
                      target_propagation=target_outputs, relative_l2=relative_l2, absolute_l2=absolute_l2,
                      raw_map_parameters=raw_maps, effective_receive_transpose_send=effective,
                      send_scale=send_scale.numpy(), receive_scale=receive_scale.numpy())
        np.savez_compressed(output / f'fold_{fold:02d}.npz', **arrays)
        audit = dict(fold=fold, test_patients=count, selection_epoch=checkpoint['best']['epoch'],
                     parameter_count=895096, baseline_acc=float(baseline_correct.mean()),
                     baseline_mean_nll=float(baseline_nll.mean()), replay_max_abs_logit_error=replay_error,
                     baseline_argmax_exact=True, prior_weights_exact=True, all_swap_edges_and_weights_exact=True,
                     all_non_target_propagation_exact=True, all_diagonal_interventions_exact=True,
                     full_scale_normalization_recomputed=True, checkpoint_sha256=sha(folder / 'best.pt'),
                     original_predictions_sha256=sha(folder / 'predictions.npz'),
                     baseline_weight_sha256=array_hash(baseline['weight'].cpu().numpy()),
                     elapsed_seconds=time.perf_counter() - started)
    finally:
        pre.remove()
        post.remove()
        del p.map_values
        sheaf_module.cooperative_operator = original_operator
    assert model_fingerprint(model) == state_before
    assert [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules()] == hook_counts_before
    assert not model.training and all(not module.training for module in model.modules())
    audit.update(unchanged_parameters_and_buffers=True, all_hooks_restored=True,
                 all_eval=True, output_sha256=sha(output / f'fold_{fold:02d}.npz'))
    dump(output / f'fold_{fold:02d}_audit.json', audit)
    del model
    return audit


def summarize(output, names, audits):
    m = len(names)
    count = sum(a['test_patients'] for a in audits)
    fold_nll, fold_margin, fold_l2, fold_acc = [], [], [], []
    changed = np.zeros((m, m), dtype=int)
    corrected, harmed = changed.copy(), changed.copy()
    all_patient_rows, union_changed, union_corrected, union_harmed = [], [], [], []
    for audit in audits:
        with np.load(output / f"fold_{audit['fold']:02d}.npz", allow_pickle=False) as a:
            fold_nll.append(a['delta_nll'].mean(-1))
            fold_margin.append(a['delta_margin'].mean(-1))
            fold_l2.append(a['relative_l2'].mean(-1))
            fold_acc.append((a['corrected'].sum(-1) - a['harmed'].sum(-1)) / len(a['labels']))
            for field, accumulator in [('changed', changed), ('corrected', corrected), ('harmed', harmed)]:
                accumulator += a[field].sum(-1)
            all_patient_rows.extend(a['heldout_row_ordinals'].tolist())
            union_changed.extend(a['changed'].any((0, 1)).tolist())
            union_corrected.extend(a['corrected'].any((0, 1)).tolist())
            union_harmed.extend(a['harmed'].any((0, 1)).tolist())
    assert len(all_patient_rows) == len(set(all_patient_rows))
    if len(audits) == 10:
        np.testing.assert_array_equal(sorted(all_patient_rows), np.arange(598))
    fold_nll, fold_margin, fold_l2, fold_acc = map(np.asarray, (fold_nll, fold_margin, fold_l2, fold_acc))
    mean_nll = fold_nll.mean(0)
    sd_nll = fold_nll.std(0, ddof=1) if len(audits) > 1 else np.zeros((m, m))
    np.savez_compressed(output / 'aggregate.npz', modality_names=np.asarray(names, dtype='U32'),
        folds=np.asarray([a['fold'] for a in audits]), fold_mean_delta_nll=fold_nll,
        fold_mean_delta_margin=fold_margin, fold_mean_relative_l2=fold_l2,
        fold_delta_acc=fold_acc, mean_delta_nll=mean_nll, sd_delta_nll=sd_nll,
        mean_delta_margin=fold_margin.mean(0), mean_relative_l2=fold_l2.mean(0),
        changed_counts=changed, corrected_counts=corrected, harmed_counts=harmed,
        mean_delta_acc=fold_acc.mean(0), pooled_delta_acc=(corrected - harmed) / count)
    cells = []
    for target in range(m):
        for donor in range(m):
            values = fold_nll[:, target, donor]
            cells.append(dict(target_content=names[target], donor_content=names[donor],
                mean_delta_nll=float(mean_nll[target, donor]), fold_sd_delta_nll=float(sd_nll[target, donor]),
                fold_min_delta_nll=float(values.min()), fold_max_delta_nll=float(values.max()),
                folds_nll_increased=int((values > 0).sum()), folds_nll_decreased=int((values < 0).sum()),
                mean_delta_margin=float(fold_margin[:, target, donor].mean()),
                mean_target_propagation_relative_l2=float(fold_l2[:, target, donor].mean()),
                changed_patients=int(changed[target, donor]), corrected_patients=int(corrected[target, donor]),
                harmed_patients=int(harmed[target, donor]),
                mean_fold_delta_acc_pp=float(100 * fold_acc[:, target, donor].mean()),
                pooled_delta_acc_pp=float(100 * (corrected[target, donor] - harmed[target, donor]) / count)))
    off = ~np.eye(m, dtype=bool)
    summary = dict(updated_at=timestamp(), folds=len(audits), sample_count=count, modalities=names,
                   aggregation='Equal mean of fold patient means; counts sum distinct held-out patients within each configuration',
                   baseline_mean_acc=float(np.mean([a['baseline_acc'] for a in audits])),
                   baseline_mean_nll=float(np.mean([a['baseline_mean_nll'] for a in audits])),
                   mean_delta_nll_matrix=mean_nll.tolist(), mean_relative_l2_matrix=fold_l2.mean(0).tolist(),
                   changed_counts=changed.tolist(), corrected_counts=corrected.tolist(), harmed_counts=harmed.tolist(),
                   unique_patients_changed_under_any_swap=int(np.sum(union_changed)),
                   unique_patients_corrected_under_any_swap=int(np.sum(union_corrected)),
                   unique_patients_harmed_under_any_swap=int(np.sum(union_harmed)),
                   off_diagonal_mean_delta_nll_range=[float(mean_nll[off].min()), float(mean_nll[off].max())],
                   cells=cells, audits=audits, no_training=True,
                   caveat='Historical test-selected exploratory checkpoints; swaps test trained-model dependence, not a from-training generalization benefit. Full maps affect self and neighbor terms; not modality importance.')
    dump(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cuda', type=int, default=-1)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    cfg, dd, data, graphs, paths = setup(cuda=args.cuda, k=10)
    historical, graph_hash = verify_inputs(cfg, dd, data, graphs, paths)
    names = list(dd['Modal_Name'])
    inputs = [Path(__file__).resolve(), ROOT / 'analyze_c_edge_content_weights.py',
              REFERENCE / 'PROTOCOL.json', REFERENCE / 'splits_and_graphs.npz']
    inputs += [ROOT / name for name in historical['code_sha256']]
    inputs += [Path(p) for p in paths]
    for fold in range(1, 11):
        inputs += [REFERENCE / 'content' / f'fold_{fold:02d}' / name for name in ('best.pt', 'predictions.npz')]
        inputs += [PREVIOUS / f'fold_{fold:02d}.npz']
    input_hashes = {str(p): sha(p) for p in dict.fromkeys(inputs)}
    output.mkdir(parents=True)
    plan = dict(created_at=timestamp(), experiment='Framework C frozen complete content-map swaps',
        task='TADPOLE AD/CN/SMCI', training_seed=0, folds=list(range(1, 11)), sample_count=598,
        modalities=names, checkpoint_selection='Historical first maximum TEST ACC; no new model selection',
        candidate_hash=graph_hash, device=str(cfg.Device), no_training=True,
        intervention='For target content m and donor d: map_values[:,m,:,:]=original_map_values[:,d,:,:]; all relations and all cohort patients',
        fixed='Input features, encoders, candidate edges, per-edge weights, non-target content maps and all trained parameters',
        normalization='Original cooperative operator recomputes both normalization factors with the donor send/receive scales',
        nll='Unweighted held-out negative log likelihood from float64 logsumexp of saved float32 logits',
        delta_nll='Swapped minus baseline; positive is worse',
        margin='True-class logit minus largest other-class logit; swapped minus baseline; negative is worse',
        relative_l2='Per patient norm(swapped target propagation-layer output - baseline target output) / max(norm(baseline target output),1e-12)',
        aggregation='Mean over held-out patients within fold, then equal mean of ten folds; fold SD descriptive only',
        counts='Each configuration sums changed/corrected/harmed over unique held-out patients across folds; union over configurations reported separately',
        all_36_combinations=True, no_result_based_cell_selection=True,
        interpretation='Dependence on learned content-map assignment. Complete S/T swaps alter both self and neighbor terms; not content importance or isolated cross-patient-message contribution. No from-training utility claim.',
        input_sha256=input_hashes)
    dump(output / 'PLAN.json', plan)
    started, audits = time.perf_counter(), []
    try:
        for fold in range(1, 11):
            audit = analyze_fold(fold, cfg, dd, data, graphs, output)
            audits.append(audit)
            summary = summarize(output, names, audits)
            dump(output / 'status.json', dict(state='running', completed_folds=fold, planned_folds=10,
                 updated_at=timestamp(), elapsed_seconds=time.perf_counter() - started))
            print(json.dumps(dict(completed_fold=fold, unique_changed=summary['unique_patients_changed_under_any_swap'],
                delta_nll_range=summary['off_diagonal_mean_delta_nll_range'],
                seconds=time.perf_counter() - started)), flush=True)
        for name, digest in input_hashes.items():
            assert sha(name) == digest, f'Changed input: {name}'
        outputs = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                   for p in sorted(output.iterdir()) if p.is_file() and p.name not in ('status.json', 'COMPLETE.json')}
        dump(output / 'COMPLETE.json', dict(state='complete', completed_at=timestamp(), folds=10,
             sample_count=598, combinations=36, no_training=True, input_sha256=input_hashes, files=outputs))
        dump(output / 'status.json', dict(state='complete', completed_folds=10, elapsed_seconds=time.perf_counter() - started,
                                         updated_at=timestamp()))
        print(json.dumps(dict(state='complete', result=str(output),
             off_diagonal_delta_nll_range=summary['off_diagonal_mean_delta_nll_range'],
             unique_changed=summary['unique_patients_changed_under_any_swap'])), flush=True)
    except BaseException as error:
        dump(output / 'status.json', dict(state='failed', completed_folds=len(audits), error=repr(error), updated_at=timestamp()))
        raise


if __name__ == '__main__':
    main()
