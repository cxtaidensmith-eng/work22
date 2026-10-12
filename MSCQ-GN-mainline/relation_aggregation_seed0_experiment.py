"""R1 shared vs R2 patient-conditioned relation aggregation, original C protocol."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (base_setup, read, state_hash, rng_state,
    assert_rng_equal, metrics, verify_manifest, verify_reference, save_predictions)
from Model.relation_sheaf import RelationSheafModel
from Model.relation_aggregation import RelationAggregationModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_relation_aggregation import summarize, write_report, PARAMETERS

OUT=ROOT/'Result/relation_aggregation_seed0_20260924'
REFERENCE=ROOT/'Result/relation_mechanism_nodecay'
MODES=('R1','R2')
FOLDS=tuple(range(1,11))
DIAGNOSTIC_EPOCHS=(1,100,500,1000)
NEW_SOURCES=('Model/relation_aggregation.py','relation_aggregation_seed0_experiment.py',
             'analyze_relation_aggregation.py','tests/test_relation_aggregation.py')


def make_model(dd,cfg,graphs,mode):
    return RelationAggregationModel(dd,cfg,graphs,mode,training_seed=0).to(cfg.Device)


def verify_sources(plan):
    verify_manifest(plan['code_sha256'])


@torch.no_grad()
def interventions(model,data,test):
    model.eval();model.capture_diagnostics=True
    layer=model.propagation
    assert layer.intervention=='normal'
    original_hash=state_hash(model);before=rng_state()
    base=model(data['Feature'])[0]
    prediction=base[test].argmax(-1);target=data['Label'][test];correct=prediction==target
    result={'normal':dict(metrics=metrics(base[test],target),changed=0,corrected=0,harmed=0)}
    diagnostic=dict(model.last_diagnostics)
    normal_weights=layer.last_weights.clone()
    try:
        for kind in ('patient_mean','uniform'):
            layer.intervention=kind
            logits=model(data['Feature'])[0][test]
            pred=logits.argmax(-1);good=pred==target
            result[kind]=dict(metrics=metrics(logits,target),changed=int((pred!=prediction).sum()),
                             corrected=int((~correct&good).sum()),harmed=int((correct&~good).sum()))
    finally:
        layer.intervention='normal'
        layer.last_weights=normal_weights
        layer.last_diagnostics=dict(diagnostic);model.last_diagnostics=dict(diagnostic)
    assert state_hash(model)==original_hash
    assert_rng_equal(before,rng_state())
    return result,diagnostic


def smoke(cfg,dd,data,graphs):
    models={};states={};initial={}
    for mode in ('original','C',*MODES):
        SET_Random(0)
        models[mode]=(RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
                      if mode=='original' else make_model(dd,cfg,graphs,mode))
        states[mode]=rng_state();assert_rng_equal(states['original'],states[mode])
        model=models[mode]
        count=sum(p.numel() for p in model.parameters())
        assert count==PARAMETERS['C' if mode=='original' else mode]
        for name,p in models['original'].state_dict().items():
            torch.testing.assert_close(p,model.state_dict()[name],atol=0,rtol=0)
        for name in ('edge','similarity','edge_group'):
            torch.testing.assert_close(getattr(models['original'].propagation,name),
                                       getattr(model.propagation,name),atol=0,rtol=0)
        initial[mode]=dict(state_sha256=state_hash(model),parameters=count)
    assert initial['original']['state_sha256']==initial['C']['state_sha256']
    assert initial['R1']['state_sha256']==initial['R2']['state_sha256']
    old=read(ROOT/'Result/alternating_cooperative_seed0_20260923/PLAN.json')
    assert initial['C']['state_sha256']==old['smoke']['initial_model_sha256']
    errors={}
    for train in (False,True):
        ref=None;ref_rng=None
        for mode,model in models.items():
            SET_Random(3701);model.train(train)
            with torch.no_grad():out=model(data['Feature'])
            if mode=='original':ref=out;ref_rng=rng_state()
            torch.testing.assert_close(ref,out,atol=2e-5,rtol=2e-5)
            assert_rng_equal(ref_rng,rng_state())
            errors[f'{mode}_{train}']=float((ref[0]-out[0]).abs().max())
    del models,model,ref,out
    torch.cuda.empty_cache()
    rows=[]
    for mode in MODES:
        SET_Random(0);model=make_model(dd,cfg,graphs,mode)
        assert state_hash(model)==initial[mode]['state_sha256']
        optimizer=relation_optimizer(model,cfg,0.)
        criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
        fit=data['Mask'][0][0]
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        score_gradients={}
        for step in range(8):
            model.train();model.capture_diagnostics=False;optimizer.zero_grad(set_to_none=True)
            logits,emb,aux=model(data['Feature'])
            loss=criterion(logits,data['Label'],fit,emb,aux)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all(),name
                if name.startswith('propagation.'):assert p.grad is not None,name
                if name.startswith(('propagation.relation_score.','propagation.relation_bias')):
                    score_gradients[name]=max(score_gradients.get(name,0.),float(p.grad.abs().max()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            model.eval();model.capture_diagnostics=step==7
            with torch.no_grad():assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        assert all(v>0 for v in score_gradients.values()),score_gradients
        trace=model.last_diagnostics
        assert trace['mean_abs_weight_deviation']>0
        assert trace['patient_weight_std']<1e-7 if mode=='R1' else trace['patient_weight_std']>1e-8
        group_manifest=optimizer_manifest(model,optimizer)
        old_groups=read(REFERENCE/'seed_0/content/fold_01/optimizer_groups.json')
        new_names=set(dict(model.named_parameters()))-set(old_groups[0]['names'])-set(old_groups[1]['names'])
        assert new_names and all(n.startswith(('propagation.relation_score.','propagation.relation_bias')) for n in new_names)
        for actual,reference in zip(group_manifest,old_groups):
            assert actual['weight_decay']==reference['weight_decay'] and actual['lr']==reference['lr']
            assert set(actual['names'])-new_names==set(reference['names'])
        rows.append(dict(mode=mode,passed=True,test_metrics_inspected=False,
            seconds_per_epoch=(time.perf_counter()-start)/8,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            diagnostics=trace,optimizer=group_manifest,score_gradient_maxima=score_gradients))
        del model,optimizer,criterion,logits,emb,aux,loss
        torch.cuda.empty_cache()
    assert rows[0]['optimizer']==rows[1]['optimizer']
    return dict(passed=True,initialization=initial,initial_output_max_errors=errors,
                common_initialization_and_rng_matched=True,smoke=rows)


def prepare():
    cfg,dd,data,graphs,hashes,original=base_setup()
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['training_seed']==0 and plan['modes']==list(MODES) and plan['data_sha256']==hashes
        for name,digest in read(OUT/'artifacts_manifest.json').items():assert sha(OUT/name)==digest
        return cfg,dd,data,graphs,plan
    check_log=ROOT/'relation_aggregation_checks.log'
    text=check_log.read_text(encoding='utf-8-sig')
    assert 'OK' in text and 'FAILED' not in text
    manifest,reference_audit=verify_reference(cfg,dd,data,graphs)
    checks=smoke(cfg,dd,data,graphs)
    old_plan=read(ROOT/'Result/alternating_cooperative_seed0_20260923/PLAN.json')
    verify_manifest(old_plan['code_sha256'])
    sources=tuple(dict.fromkeys((*old_plan['code_sha256'],*NEW_SOURCES)))
    plan=dict(experiment='C relation aggregation: shared versus patient-conditioned',training_seed=0,split_seed=0,
        modes=list(MODES),folds=list(FOLDS),epochs=1000,planned_runs=20,reused_C_runs=10,
        class_names=dd['Class_Names'],modalities=dd['Modal_Name'],samples=dd['Sample_Num'],
        original_config=original['original_config'],class_weights=dd['Label_Weight'].cpu().tolist(),
        data_sha256=hashes,candidate_hash=original['candidate_hash'],transductive=True,
        checkpoint_selection=original['checkpoint_selection'],parameters=PARAMETERS,
        architecture=dict(C='Original mean over six relation-source cooperative updates',
            R1='Shared 192->16 GELU->1 scorer, output has no bias; 6x6 relation/content bias. Softmax(mean cohort scores) broadcast to all patients.',
            R2='Exactly same scorer and parameters, softmax per patient over six relation sources.',
            normalized_inputs='Affine-free LayerNorm of graph input-normalized content token z and each cooperative update; concatenate in that order.',
            reduction='Weight raw cooperative updates before original GELU/output projection/residual/LayerNorm; no graph change.'),
        initialization='Fresh C seed0 per fold; identical common parameters. Extra scorer seed9101 protected by CPU/all-CUDA RNG fork. Last scoring weight and relation bias zero => initial uniform weights. No trained warm start.',
        optimizer=old_plan['optimizer'],new_parameter_decay='All added aggregation parameters are graph-only and use weight_decay=0 under the existing rule.',
        objective=old_plan['objective'],run_order='For each original fold, R1 then R2; 20 new full fits',
        interventions='At fixed best/last checkpoints replace post-softmax weights by cohort mean or uniform; no retraining or re-selection. Full alpha tensors saved.',
        primary='Same seed/fold best ACC: R2-C and R2-R1; R1-C tests shared preference plus added capacity.',
        secondary='Macro F1, AUC, per-class recall, epoch1000, last100 mean, paired corrections/errors.',
        scope='No label input to scorer, new loss, early stop, new candidate graph, extra seeds, tuning or automatic follow-up. Cohort-score averaging retains existing transductive scope.',
        code_sha256={name:sha(ROOT/name) for name in sources},reference_manifest=manifest,
        initialization_checks=checks,reference_audit=reference_audit,
        environment=dict(executable=sys.executable,python=sys.version,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    OUT.mkdir(parents=True,exist_ok=False)
    for source in sources:
        destination=OUT/'source_snapshot'/source;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/source).read_bytes())
    (OUT/'data_splits_and_graphs.npz').write_bytes((REFERENCE/'seed_0/splits_and_graphs.npz').read_bytes())
    (OUT/'implementation_checks.log').write_bytes(check_log.read_bytes())
    dump(OUT/'PLAN.json',plan)
    dump(OUT/'artifacts_manifest.json',{name:sha(OUT/name) for name in
         ('data_splits_and_graphs.npz','implementation_checks.log')})
    summarize(OUT,plan)
    dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp(),planned_runs=20))
    return cfg,dd,data,graphs,plan


def save_checkpoint(folder,stage,model,epoch,score,fold,mode,plan):
    temp=folder/f'{stage}.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,metrics=score,seed=0,fold=fold,
                    mode=mode,protocol_sha256=sha(OUT/'PLAN.json')),temp)
    temp.replace(folder/f'{stage}.pt')


def train_arm(cfg, dd, data, graphs, plan, fold, mode):
    folder = OUT/f'seed_0/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists():
        return
    verify_sources(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(0)
    model = make_model(dd, cfg, graphs, mode)
    initial_hash = state_hash(model)
    assert initial_hash == plan['initialization_checks']['initialization'][mode]['state_sha256']
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold-1]
    best_correct, best, last, history = -1, None, None, []
    diagnostics = {}
    parameters = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','train_loss','test_loss','test_acc','lr','seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.capture_diagnostics = False
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} fold {fold} epoch {epoch}: nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            model.capture_diagnostics = epoch in DIAGNOSTIC_EPOCHS
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
            if model.capture_diagnostics:
                diagnostics[str(epoch)] = model.last_diagnostics
                dump(folder/'diagnostics.json', diagnostics)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                       test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'],
                       seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'best', model, epoch, best, fold, mode, plan)
                save_predictions(folder/'predictions.npz', logits, y, test, dd['Index'])
                dump(folder/'best.json', best)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                              fold=fold, seed=0, epochs=1000, planned_runs=20,
                              parameters=parameters, best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                print(json.dumps(status), flush=True)
            if epoch == 1000:
                last = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'last', model, epoch, last, fold, mode, plan)
                save_predictions(folder/'last_predictions.npz', logits, y, test, dd['Index'])
    record = dict(mode=mode, fold=fold, seed=0, parameters=parameters, best=best, last=last,
                  initial_model_sha256=initial_hash, mean_last100_acc=float(np.mean(history[-100:])),
                  best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
                  completed_at=timestamp(), seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache()
    summarize(OUT, plan)


def audit_completed(cfg,dd,data,graphs,plan):
    verify_sources(plan);verify_manifest(plan['reference_manifest'])
    for name,digest in read(OUT/'artifacts_manifest.json').items():assert sha(OUT/name)==digest
    rows=[]
    for mode in MODES:
        SET_Random(0);model=make_model(dd,cfg,graphs,mode).eval()
        for fold in FOLDS:
            folder=OUT/f'seed_0/{mode}/fold_{fold:02d}'
            record=read(folder/'complete.json')
            assert (record['mode'],record['seed'],record['fold'])==(mode,0,fold)
            assert record['parameters']==PARAMETERS[mode]
            assert record['initial_model_sha256']==plan['initialization_checks']['initialization'][mode]['state_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:epochs=list(csv.DictReader(stream))
            with (REFERENCE/f'seed_0/content/fold_{fold:02d}/epochs.csv').open(encoding='utf-8') as stream:original=list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs]==list(range(1,1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1==record['best']['epoch']
            np.testing.assert_array_equal([float(r['lr']) for r in epochs],[float(r['lr']) for r in original])
            test=data['Mask'][fold-1][1]
            for stage,filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                assert sha(folder/f'{stage}.pt')==record[f'{stage}_checkpoint_sha256']
                cp=torch.load(folder/f'{stage}.pt',map_location=cfg.Device,weights_only=False)
                assert cp['protocol_sha256']==sha(OUT/'PLAN.json')
                assert (cp['mode'],cp['seed'],cp['fold'])==(mode,0,fold)
                assert cp['epoch']==record[stage]['epoch'] and cp['metrics']==record[stage]
                model.load_state_dict(cp['state_dict'],strict=True)
                model.capture_diagnostics=True
                with torch.no_grad():logits=model(data['Feature'])[0]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'],np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'],data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'],logits[test].cpu().numpy(),atol=2e-5,rtol=2e-5)
                    np.testing.assert_array_equal(saved['logits'].argmax(-1),logits[test].argmax(-1).cpu().numpy())
                    error=float(np.max(np.abs(saved['logits']-logits[test].cpu().numpy())))
                actual=metrics(logits[test],data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','nll','brier'):assert abs(actual[key]-record[stage][key])<2e-5
                np.testing.assert_array_equal(actual['confusion_matrix'],record[stage]['confusion_matrix'])
                result,diagnostic=interventions(model,data,test)
                weights=model.propagation.last_weights.cpu().numpy()
                assert weights.shape==(dd['Sample_Num'],len(dd['Modal_Name']),len(dd['Modal_Name']))
                assert np.isfinite(weights).all() and (weights>=0).all()
                np.testing.assert_allclose(weights.sum(1),1.,atol=2e-6)
                if mode=='R1':np.testing.assert_allclose(weights,np.broadcast_to(weights[:1],weights.shape),atol=1e-7)
                np.savez_compressed(folder/f'weights_{stage}.npz',weights=weights,row_index=np.asarray(dd['Index']),
                                    test_mask=test.cpu().numpy(),modalities=np.asarray(dd['Modal_Name']))
                rows.append(dict(mode=mode,seed=0,fold=fold,stage=stage,max_abs_replay_error=error,
                                 interventions=result,diagnostics=diagnostic,weights_sha256=sha(folder/f'weights_{stage}.npz')))
            diagnostic=read(folder/'diagnostics.json')
            assert sorted(map(int,diagnostic))==list(DIAGNOSTIC_EPOCHS)
            assert all(np.isfinite(list(v.values())).all() for v in diagnostic.values())
            assert sum(p.numel() for p in model.parameters())==PARAMETERS[mode]
            assert optimizer_manifest(model,relation_optimizer(model,cfg,0.))==read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result=dict(passed=True,verified_runs=20,verified_checkpoints=40,records=rows,
        aggregation_axes='patient, relation source, content modality',
        reference_best_checkpoints_replayed_at_preparation=10,
        same_original_lr_schedule=True,same_original_source_and_reference_hashes=True)
    dump(OUT/'audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,graphs,plan=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared',planned_runs=20,parameters=plan['parameters'],
            checks=plan['initialization_checks'])),flush=True)
        return
    if (OUT/'COMPLETE.json').exists():
        print('Already complete; retained results unchanged.',flush=True);return
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    try:
        for fold in FOLDS:
            for mode in MODES:train_arm(cfg,dd,data,graphs,plan,fold,mode)
        dump(OUT/'status.json',dict(state='auditing',pid=os.getpid(),updated_at=timestamp()))
        audit=audit_completed(cfg,dd,data,graphs,plan)
        summary=write_report(OUT,plan,audit)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp(),completed_runs=20))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',pid=os.getpid(),updated_at=timestamp(),error=repr(error)))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
