"""Label-informed diagnostic ceiling for 64 fixed modal-reference masks.

Reuses the always-open teachers fitted on inner training partitions. All
selection uses INNER holdout labels only. Nothing is trained, no outer-test
metrics are computed, and oracle scores are not predictive performance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relation_experiment import ROOT, setup, dump, sha, timestamp
from reference_selection_experiment import read
from Utils.reference_selection import coalition_masks, inner_splits
from Model.reference_selection import ReferenceSelectionModel


def best_masks(losses, masks, tolerance=1e-7):
    """Prefer retaining more references for numerical ties, then larger ID."""
    losses=np.asarray(losses,dtype=np.float64)
    if losses.ndim==1: losses=losses[:,None]
    if len(losses)!=len(masks) or not np.isfinite(losses).all():
        raise ValueError('Invalid coalition losses')
    order=np.lexsort((-np.arange(len(masks)),-masks.sum(1)))
    eligible=losses <= losses.min(0,keepdims=True)+tolerance
    return order[eligible[order].argmax(0)]


def describe(values):
    values=np.asarray(values,dtype=np.float64)
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),
                min=float(values.min()),max=float(values.max()))


def analyze_arrays(logits,losses,labels,masks):
    logits=np.asarray(logits); losses=np.asarray(losses,dtype=np.float64); labels=np.asarray(labels)
    k,n,c=logits.shape
    assert losses.shape==(k,n) and masks.shape[0]==k and labels.shape==(n,)
    assert np.array_equal(masks,coalition_masks(masks.shape[1]))
    if not np.isfinite(logits).all() or not np.isfinite(losses).all():
        raise ValueError('Nonfinite diagnostic input')
    pred=logits.argmax(-1); correct=pred==labels[None]
    means=losses.mean(1)
    global_id=int(best_masks(means,masks)[0])
    individual=best_masks(losses,masks)
    ids=np.arange(n)
    individual_correct=correct[individual,ids]
    best_acc=correct.mean(1).max()
    acc_ties=correct.mean(1)>=best_acc-1e-12
    # Secondary ACC oracle has its own criterion, unlike the primary NLL oracle.
    candidates=np.flatnonzero(acc_ties)
    acc_id=int(candidates[best_masks(means[candidates],masks[candidates])[0]])
    any_correct=correct.any(0)
    all_id=k-1
    def fixed(j):
        return dict(mask_id=int(j),mask=masks[j].astype(int).tolist(),correct=int(correct[j].sum()),
                    acc=float(correct[j].mean()),nll=float(means[j]))
    individual_loss=losses[individual,ids]
    gain_all=losses[all_id]-individual_loss
    gain_global=losses[global_id]-individual_loss
    result=dict(patients=n,all_open=fixed(all_id),all_closed=fixed(0),
        global_nll_oracle=fixed(global_id),global_acc_oracle=fixed(acc_id),
        individual_nll_oracle=dict(acc=float(individual_correct.mean()),correct=int(individual_correct.sum()),
            nll=float(individual_loss.mean()),
            corrected_vs_all=int((~correct[all_id]&individual_correct).sum()),
            harmed_vs_all=int((correct[all_id]&~individual_correct).sum()),
            corrected_vs_global=int((~correct[global_id]&individual_correct).sum()),
            harmed_vs_global=int((correct[global_id]&~individual_correct).sum()),
            nll_gain_over_all=describe(gain_all),nll_gain_over_global=describe(gain_global),
            nll_gain_over_global_above_001=int((gain_global>.001).sum()),
            nll_gain_over_global_above_01=int((gain_global>.01).sum()),
            chosen_mask_histogram={str(int(j)):int((individual==j).sum()) for j in np.unique(individual)}),
        individual_acc_oracle=dict(correct=int(any_correct.sum()),acc=float(any_correct.mean()),
            additional_correct_over_all=int((any_correct&~correct[all_id]).sum()),
            additional_correct_over_global_acc=int((any_correct&~correct[acc_id]).sum()),
            definition='Correct if ANY of 64 masks predicts the true class; uses patient labels, unattainable oracle'),
        prediction_changes=dict(any_change_vs_all=int((pred!=pred[all_id]).any(0).sum()),
            invariant_count=int((pred==pred[all_id]).all(0).sum()),
            all_open_errors=int((~correct[all_id]).sum()),
            uncorrectable_by_any_mask=int((~any_correct).sum())),
        ambiguity=dict(best_to_second_nll_gap=describe(np.sort(losses,axis=0)[1]-np.sort(losses,axis=0)[0]),
            masks_within_001_of_best=describe((losses<=losses.min(0)+.001).sum(0)),
            masks_within_01_of_best=describe((losses<=losses.min(0)+.01).sum(0))))
    assert individual_loss.mean()<=means[global_id]+1.01e-7
    assert any_correct.sum()>=correct[acc_id].sum()
    return result,individual


def aggregate(records):
    total=sum(r['patients'] for r in records)
    keys=('all_open','all_closed','global_nll_oracle','global_acc_oracle','individual_nll_oracle','individual_acc_oracle')
    result=dict(patients=total,groups=len(records),groups_are_distinct_patients=True)
    for key in keys:
        result[key]=dict(correct=sum(r[key]['correct'] for r in records),
                        acc=sum(r[key]['correct'] for r in records)/total)
        if 'nll' in records[0][key]:
            result[key]['nll']=sum(r[key]['nll']*r['patients'] for r in records)/total
    result['extra_correct_over_global_acc_oracle']=sum(r['individual_acc_oracle']['additional_correct_over_global_acc'] for r in records)
    result['any_changed_prediction_count']=sum(r['prediction_changes']['any_change_vs_all'] for r in records)
    result['individual_over_global_nll_delta']=result['global_nll_oracle']['nll']-result['individual_nll_oracle']['nll']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Result/reference_selection_space')
    parser.add_argument('--source',default='Result/reference_signal_diagnosis')
    args=parser.parse_args()
    out,source=ROOT/args.output,ROOT/args.source
    if out.exists(): raise FileExistsError(out)
    original=read(ROOT/'Result/reference_selection/PROTOCOL.json')
    source_plan=read(source/'PLAN.json')
    assert read(source/'status.json')['state']=='complete'
    assert source_plan['outer_fold']==1 and source_plan['training_seed']==0 and source_plan['epochs']==1000
    for path,digest in original['code_sha256'].items():
        assert sha(ROOT/path)==digest, f'Old source changed: {path}'
    cfg,dd,data,graphs,paths=setup(k=original['candidate_k'])
    assert {Path(p).name:sha(p) for p in paths}==original['data_sha256']
    partitions=inner_splits(data['Label'],*data['Mask'][0],31010,2)
    checkpoints={str(inner):sha(source/f'inner_{inner}/always_open/last.pt') for inner in (0,1)}
    plan=dict(created_at=timestamp(),source_plan_sha256=sha(source/'PLAN.json'),checkpoint_sha256=checkpoints,
        analysis_source_sha256=sha(Path(__file__)),fixed_teacher_epochs=1000,outer_fold=1,
        inner_partitions=[0,1],teacher_policy='always_open',combination_count=64,
        primary_selection='Unweighted true-label NLL: best shared mask per holdout and best mask per patient',
        secondary_selection='Best shared ACC mask and any-correct patientwise ACC oracle; separate from NLL criterion',
        numerical_ties='Absolute NLL tolerance 1e-7; retain more references, then larger bit mask ID',
        near_tie_reporting=[.001,.01],
        additional_check='Apply selected patient-specific masks simultaneously and reproduce stitched predictions',
        temporal_check='Apply epoch-200 NLL-selected masks to epoch-1000 losses; same patients, not an independent generalization test',
        labels='Only inner holdout labels for oracle diagnostics; no outer-test metrics or selection',
        scope='Two complementary internal holdouts of one outer fold, not two seeds on the same patients',
        caveat='Label-informed optimistic diagnostic bounds on 64 binary masks; no fitted selector or deployable result')
    out.mkdir(parents=True)
    dump(out/'PLAN.json',plan)
    records=[]; seen=[]
    try:
        with torch.no_grad():
            for inner,(train_np,held_np) in enumerate(partitions):
                folder=source/f'inner_{inner}/always_open'
                result_folder=out/f'inner_{inner}'; result_folder.mkdir()
                teacher=read(folder/'complete.json')
                held=torch.as_tensor(held_np,device=cfg.Device)
                assert not np.any(held_np & data['Mask'][0][1].cpu().numpy())
                assert not np.any(train_np & held_np)
                model=ReferenceSelectionModel(dd,cfg,graphs,teacher['teacher_seed'],learned=False).to(cfg.Device).eval()
                checkpoint=torch.load(folder/'last.pt',map_location=cfg.Device,weights_only=False)
                assert checkpoint['epoch']==1000 and checkpoint['policy']=='always_open' and checkpoint['inner']==inner
                model.load_state_dict(checkpoint['state_dict'])
                masks=coalition_masks(len(dd['Modal_Name']))
                logits=np.stack([model(data['Feature'],mask[None])[0][held].cpu().numpy() for mask in masks])
                y=data['Label'][held].cpu().numpy()
                losses=F.cross_entropy(torch.from_numpy(logits).reshape(-1,dd['Class_Num']),
                    torch.from_numpy(np.tile(y,len(masks))),reduction='none').reshape(len(masks),-1).numpy()
                with np.load(folder/'coalitions_epoch_1000.npz') as saved:
                    np.testing.assert_array_equal(saved['held_mask'],held_np)
                    np.testing.assert_array_equal(saved['masks'],masks)
                    np.testing.assert_allclose(losses,saved['losses'],atol=2e-5,rtol=2e-5)
                result,choices=analyze_arrays(logits,losses,y,masks)
                patient_masks=np.ones((dd['Sample_Num'],len(dd['Modal_Name'])),dtype=np.float32)
                patient_masks[held_np]=masks[choices]
                simultaneous=model(data['Feature'],patient_masks)[0][held].cpu().numpy()
                np.testing.assert_allclose(simultaneous,logits[choices,np.arange(len(y))],atol=2e-5,rtol=2e-5)
                temporal={}
                with np.load(folder/'coalitions_epoch_0200.npz') as old:
                    early_choices=best_masks(old['losses'],masks)
                    transferred=losses[early_choices,np.arange(len(y))]
                    early_gain=old['losses'][-1]-old['losses'][early_choices,np.arange(len(y))]
                    temporal=dict(exact_mask_agreement=float((early_choices==choices).mean()),
                        mean_bit_agreement=float((masks[early_choices]==masks[choices]).mean()),
                        nll_when_using_early_choices=float(transferred.mean()),
                        nll_improvement_vs_all_open=float((losses[-1]-transferred).mean()),
                        patients_with_early_gain_above_01=int((early_gain>.01).sum()),
                        early_choices_are_on_same_patients=True)
                result.update(inner=inner,teacher_seed=teacher['teacher_seed'],modalities=dd['Modal_Name'],
                              simultaneous_masks_verified=True,temporal_diagnostic=temporal)
                np.savez_compressed(result_folder/'predictions.npz',logits=logits,losses=losses,labels=y,
                    source_row=np.asarray(dd['Index'])[held_np],held_mask=held_np,train_mask=train_np,
                    masks=masks,individual_nll_choices=choices)
                dump(result_folder/'analysis.json',result)
                records.append(result); seen.append(held_np)
                print(json.dumps(dict(event='inner_complete',inner=inner,all_open=result['all_open'],
                    global_nll=result['global_nll_oracle'],individual_nll=result['individual_nll_oracle']['acc'],
                    global_acc=result['global_acc_oracle'],individual_acc=result['individual_acc_oracle'])),flush=True)
        np.testing.assert_array_equal(np.stack(seen).sum(0),data['Mask'][0][0].cpu().numpy().astype(int))
        summary=dict(completed_at=timestamp(),aggregate=aggregate(records),groups=records,
            outer_test_unused=True,scope=plan['scope'],caveat=plan['caveat'])
        dump(out/'analysis.json',summary)
        dump(out/'audit.json',dict(passed=True,checkpoints_replayed=2,coalitions_per_checkpoint=64,
            heldout_masks_verified=True,outer_test_excluded=True,simultaneous_patient_masks_verified=True,
            holdout_rows_disjoint=True,all_teacher_checkpoints_fixed_before_analysis=True))
        dump(out/'COMPLETE.json',summary['aggregate'])
        print(json.dumps(summary['aggregate']),flush=True)
    except BaseException as error:
        dump(out/'FAILED.json',dict(error=repr(error),at=timestamp()))
        raise


if __name__=='__main__':
    main()
