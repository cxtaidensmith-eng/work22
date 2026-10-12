"""Prespecified replication of the binary modality-reference oracle diagnosis.

Two new outer folds x two training seeds x two complementary inner holdouts.
Internal partitions are identical across training seeds. The outer test labels
never enter training, monitoring, mask selection, or reported metrics.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp
from reference_selection_experiment import read, model_hash, optim_and_loss, step
from reference_selection_space import analyze_arrays, aggregate
from Utils import SET_Random
from Utils.reference_selection import inner_splits, coalition_masks, training_class_weights, array_hash
from Model.reference_selection import ReferenceSelectionModel


FOLDS = (2, 3)
SEEDS = (0, 1)
EPOCHS = 1000


def training_inputs(y, fit, classes):
    # Supervised objectives and their class weights see fitting labels only.
    return torch.where(fit, y, torch.zeros_like(y)), training_class_weights(y, fit, classes)


@torch.no_grad()
def evaluate(model, x, y, dd, train_np, held_np, folder):
    model.eval()
    held = torch.as_tensor(held_np, device=x.device)
    masks = coalition_masks(len(dd['Modal_Name']))
    logits = np.stack([model(x, mask[None])[0][held].cpu().numpy() for mask in masks])
    labels = y[held].cpu().numpy()
    losses = F.cross_entropy(torch.from_numpy(logits).reshape(-1, dd['Class_Num']),
        torch.from_numpy(np.tile(labels, len(masks))), reduction='none').reshape(len(masks), -1).numpy()
    result, choices = analyze_arrays(logits, losses, labels, masks)
    patient_masks = np.ones((len(y), len(dd['Modal_Name'])), dtype=np.float32)
    patient_masks[held_np] = masks[choices]
    simultaneous = model(x, patient_masks)[0][held].cpu().numpy()
    np.testing.assert_allclose(simultaneous, logits[choices, np.arange(len(labels))], atol=2e-5, rtol=2e-5)
    np.savez_compressed(folder/'predictions.npz', logits=logits, losses=losses,
        labels=labels, masks=masks, source_row=np.asarray(dd['Index'])[held_np],
        held_mask=held_np, train_mask=train_np, individual_nll_choices=choices)
    result['simultaneous_patient_masks_verified'] = True
    return result


def fit_teacher(out, folder, cfg, dd, data, graphs, fold, seed, inner, train_np, held_np):
    folder.mkdir(parents=True, exist_ok=True)
    teacher_seed = 52001 + seed*100 + fold*2 + inner
    SET_Random(teacher_seed)
    model = ReferenceSelectionModel(dd, cfg, graphs, teacher_seed, learned=False).to(cfg.Device)
    initial_hash = model_hash(model)
    x, y = data['Feature'], data['Label']
    fit = torch.as_tensor(train_np, device=cfg.Device)
    teacher_y, weights = training_inputs(y, fit, dd['Class_Num'])
    criterion, optimizer, scheduler = optim_and_loss(model, cfg, dict(dd, Label_Weight=weights), folder)
    started = time.perf_counter()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'lr', 'seconds'])
        for epoch in range(1, EPOCHS+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, teacher_y, fit, embeddings, auxiliary)
            step(model, optimizer, loss, cfg)
            writer.writerow([epoch, float(loss.detach()), optimizer.param_groups[0]['lr'], time.perf_counter()-started])
            if scheduler:
                scheduler.step()
            if epoch == 1 or epoch % 100 == 0:
                stream.flush()
                status = dict(state='training', updated_at=timestamp(), fold=fold, seed=seed, inner=inner,
                    epoch=epoch, train_loss=float(loss.detach()), seconds=time.perf_counter()-started)
                dump(out/'status.json', status)
                print(json.dumps(status), flush=True)
    torch.save(dict(state_dict=model.state_dict(), teacher_seed=teacher_seed, epoch=EPOCHS,
        policy='always_open', outer_fold=fold, training_seed=seed, inner=inner), folder/'last.pt')
    trained_hash = model_hash(model)
    checkpoint = torch.load(folder/'last.pt', map_location=cfg.Device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    assert model_hash(model) == trained_hash
    result = evaluate(model, x, y, dd, train_np, held_np, folder)
    result.update(outer_fold=fold, training_seed=seed, inner=inner, teacher_seed=teacher_seed,
        inner_split_seed=31009+fold, epochs=EPOCHS, teacher_policy='always_open',
        initial_state_sha256=initial_hash, trained_state_sha256=trained_hash,
        checkpoint_sha256=sha(folder/'last.pt'), predictions_sha256=sha(folder/'predictions.npz'),
        fit_mask_sha256=array_hash(train_np), held_mask_sha256=array_hash(held_np),
        seconds=time.perf_counter()-started, completed_at=timestamp())
    dump(folder/'complete.json', result)
    print(json.dumps(dict(event='teacher_complete', fold=fold, seed=seed, inner=inner,
        all_open=result['all_open']['correct'], shared_acc=result['global_acc_oracle']['correct'],
        individual_acc=result['individual_acc_oracle']['correct'], patients=result['patients'])), flush=True)
    return result


def summarize(out, data):
    cells = []
    for fold in FOLDS:
        for seed in SEEDS:
            base = out/f'fold_{fold:02d}'/f'seed_{seed}'
            if not all((base/f'inner_{j}'/'complete.json').exists() for j in (0, 1)):
                continue
            records = [read(base/f'inner_{j}'/'complete.json') for j in (0, 1)]
            arrays = []
            for j, record in enumerate(records):
                folder = base/f'inner_{j}'
                assert sha(folder/'last.pt') == record['checkpoint_sha256']
                assert sha(folder/'predictions.npz') == record['predictions_sha256']
                with np.load(folder/'predictions.npz') as saved:
                    arrays.append({key: saved[key].copy() for key in saved.files})
            outer_fit, outer_test = [v.cpu().numpy() for v in data['Mask'][fold-1]]
            np.testing.assert_array_equal(sum(a['held_mask'].astype(int) for a in arrays), outer_fit.astype(int))
            assert not any(np.any(a['held_mask'] & outer_test) or np.any(a['train_mask'] & outer_test)
                           or np.any(a['held_mask'] & a['train_mask']) for a in arrays)
            pooled, _ = analyze_arrays(np.concatenate([a['logits'] for a in arrays], axis=1),
                np.concatenate([a['losses'] for a in arrays], axis=1),
                np.concatenate([a['labels'] for a in arrays]), arrays[0]['masks'])
            cell = dict(outer_fold=fold, training_seed=seed, teachers=records,
                per_teacher_shared=aggregate(records), one_shared_mask_across_both_teachers=pooled)
            dump(base/'analysis.json', cell)
            cells.append(cell)
    result = dict(updated_at=timestamp(), completed_cells=len(cells), planned_cells=4, cells=cells,
        labels='Inner holdout labels only; all best-mask results are label-informed oracle diagnostics',
        repeated_patients='Cells overlap in patients and seeds; do not treat their pooled rows as independent patients')
    if len(cells) == 4:
        for fold in FOLDS:
            paired = [c for c in cells if c['outer_fold'] == fold]
            for inner in (0, 1):
                a, b = (c['teachers'][inner] for c in paired)
                assert a['fit_mask_sha256'] == b['fit_mask_sha256']
                assert a['held_mask_sha256'] == b['held_mask_sha256']
                assert a['initial_state_sha256'] != b['initial_state_sha256']
        pooled = [c['one_shared_mask_across_both_teachers'] for c in cells]
        result['descriptive_summary'] = dict(
            individual_extra_correct_over_shared_by_cell=[r['individual_acc_oracle']['additional_correct_over_global_acc'] for r in pooled],
            individual_extra_acc_pp_over_shared_by_cell=[100*(r['individual_acc_oracle']['acc']-r['global_acc_oracle']['acc']) for r in pooled],
            per_teacher_extra_correct_by_cell=[c['per_teacher_shared']['extra_correct_over_global_acc_oracle'] for c in cells],
            invariant_fraction_by_cell=[r['prediction_changes']['invariant_count']/r['patients'] for r in pooled],
            individual_nll_gain_over_shared_by_cell=[r['global_nll_oracle']['nll']-r['individual_nll_oracle']['nll'] for r in pooled],
            all_open_errors_by_cell=[r['prediction_changes']['all_open_errors'] for r in pooled],
            errors_uncorrectable_by_any_mask_by_cell=[r['prediction_changes']['uncorrectable_by_any_mask'] for r in pooled])
    dump(out/'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='Result/reference_selection_replication')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT/args.output
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    source = ROOT/'Result/reference_selection/PROTOCOL.json'
    protocol = read(source)
    for path, digest in protocol['code_sha256'].items():
        assert sha(ROOT/path) == digest, f'Historical source changed: {path}'
    cfg, dd, data, graphs, paths = setup(k=protocol['candidate_k'])
    assert {Path(p).name: sha(p) for p in paths} == protocol['data_sha256']
    assert all(getattr(cfg, k) == v for k, v in protocol['original_config'].items())
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest() == protocol['candidate_hash']
    splits = {fold: inner_splits(data['Label'], *data['Mask'][fold-1], 31009+fold, 2) for fold in FOLDS}
    plan = dict(experiment='Prespecified replication: selection space of C binary content gates',
        folds=list(FOLDS), training_seeds=list(SEEDS), teachers=8, epochs=EPOCHS,
        inner_split_seeds={str(f): 31009+f for f in FOLDS}, inner_partitions=[0, 1],
        split_policy='Same internal partitions across training seeds; only initialization and training RNG differ',
        teacher_seed_formula='52001 + 100*training_seed + 2*outer_fold + inner',
        policy='always_open', checkpoint_selection='Fixed epoch 1000; no holdout or outer-test selection',
        data_and_main_protocol='Original processed cohort, original ten outer folds, model, optimizer and losses retained',
        labels='Fit labels only for loss/class weights; inner held labels only after training for oracle analysis; no outer-test metrics',
        primary='Minimum true-label unweighted NLL: all-open, shared best, individual best',
        secondary='ACC best shared mask versus individual any-correct ceiling; not confused with NLL choice',
        shared_comparisons=['Best shared mask for each teacher', 'One shared mask across both complementary teachers per fold/seed'],
        numerical_ties='NLL tolerance 1e-7, prefer more open modalities then larger mask ID',
        aggregation='Four fold/seed cells reported separately; repeated patients are not independent samples',
        interpretation='An optimistic bound within 64 binary post-aggregation content masks; not a learned selector or a bound on all future models',
        followup='Assess reproducibility of extra individual ACC space and NLL space; no hyperparameter or model search in this replication',
        protocol_sha256=sha(source), runner_sha256=sha(Path(__file__)),
        analysis_helper_sha256=sha(ROOT/'reference_selection_space.py'),
        split_hashes={str(f): [dict(fit=array_hash(a), held=array_hash(b)) for a,b in pairs] for f,pairs in splits.items()})
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        assert read(out/'PLAN.json') == plan, 'Cannot change a frozen replication plan'
    else:
        dump(out/'PLAN.json', plan)
        dump(out/'STARTED.json', dict(started_at=timestamp()))
    try:
        for fold in FOLDS:
            for seed in SEEDS:
                for inner, (train_np, held_np) in enumerate(splits[fold]):
                    folder = out/f'fold_{fold:02d}'/f'seed_{seed}'/f'inner_{inner}'
                    if not (folder/'complete.json').exists():
                        fit_teacher(out, folder, cfg, dd, data, graphs, fold, seed, inner, train_np, held_np)
                    summarize(out, data)
        result = summarize(out, data)
        assert result['completed_cells'] == 4
        dump(out/'audit.json', dict(passed=True, completed_teachers=8, original_sources_and_data_verified=True,
            original_candidate_graph_verified=True, checkpoints_saved_and_reloaded=True,
            outer_test_excluded=True, internal_holdouts_complementary=True,
            same_inner_partitions_across_seeds=True, different_initialization_across_seeds=True,
            simultaneous_patient_masks_verified=True, fixed_epoch_without_selection=True))
        dump(out/'COMPLETE.json', result['descriptive_summary'])
        dump(out/'status.json', dict(state='complete', updated_at=timestamp(), completed_teachers=8))
        print(json.dumps(result['descriptive_summary']), flush=True)
    except BaseException as error:
        dump(out/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
