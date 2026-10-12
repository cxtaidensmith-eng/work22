"""C branch fusion: concat vs near-equal-capacity sum, original seed-0 protocol."""
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
from Model.branch_fusion import BranchFusionModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_branch_fusion import summarize, write_report, PARAMETERS

OUT=ROOT/'Result/branch_fusion_seed0_20260924'
REFERENCE=ROOT/'Result/relation_mechanism_nodecay'
MODES=('concat','wide')
FOLDS=tuple(range(1,11))
DIAGNOSTIC_EPOCHS=(1,100,500,1000)
NEW_SOURCES=('Model/branch_fusion.py','branch_fusion_seed0_experiment.py',
             'analyze_branch_fusion.py','tests/test_branch_fusion.py')


def make_model(dd,cfg,graphs,mode):
    return BranchFusionModel(dd,cfg,graphs,mode,training_seed=0).to(cfg.Device)


def verify_sources(plan):
    verify_manifest(plan['code_sha256'])


@torch.no_grad()
def interventions(model,data,test):
    model.eval();model.capture_diagnostics=True
    original_hash=state_hash(model);before=rng_state()
    base=model(data['Feature'])[0]
    prediction=base[test].argmax(-1);target=data['Label'][test]
    correct=prediction==target
    result={'both':dict(metrics=metrics(base[test],target),changed=0,corrected=0,harmed=0)}
    diagnostic=dict(model.last_diagnostics)
    for branch in ('semantic_only','own_only'):
        logits=model(data['Feature'],branch=branch)[0][test]
        pred=logits.argmax(-1);good=pred==target
        result[branch]=dict(metrics=metrics(logits,target),changed=int((pred!=prediction).sum()),
                           corrected=int((~correct&good).sum()),harmed=int((correct&~good).sum()))
    assert state_hash(model)==original_hash
    assert_rng_equal(before,rng_state())
    return result,diagnostic


def smoke(cfg,dd,data,graphs):
    models={};states={};initial={}
    for mode in ('original','C',*MODES):
        SET_Random(0)
        models[mode]=(RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).to(cfg.Device)
                      if mode=='original' else make_model(dd,cfg,graphs,mode))
        states[mode]=rng_state()
        assert_rng_equal(states['original'],states[mode])
        model=models[mode]
        count=sum(p.numel() for p in model.parameters())
        assert count==PARAMETERS['C' if mode=='original' else mode]
        for name,p in models['original'].state_dict().items():
            if mode in ('original','C') or not name.startswith('classifier.'):
                torch.testing.assert_close(p,model.state_dict()[name],atol=0,rtol=0)
        initial[mode]=dict(state_sha256=state_hash(model),parameters=count)
    assert initial['original']['state_sha256']==initial['C']['state_sha256']
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
        incoming_before=model.classifier[0].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(8):
            model.train();model.capture_diagnostics=False;optimizer.zero_grad(set_to_none=True)
            logits,emb,aux=model(data['Feature'])
            loss=criterion(logits,data['Label'],fit,emb,aux)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all(),name
                if name.startswith(('classifier.','propagation.')):assert p.grad is not None,name
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            model.eval();model.capture_diagnostics=step==7
            with torch.no_grad():assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        if mode=='wide':
            assert (model.classifier[0].weight[48:]-incoming_before[48:]).abs().max()>0
            assert model.classifier[3].weight[:,48:].abs().max()>0
        else:assert model.last_diagnostics['relative_head_split_difference']>0
        rows.append(dict(mode=mode,passed=True,test_metrics_inspected=False,
            seconds_per_epoch=(time.perf_counter()-start)/8,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            diagnostics=model.last_diagnostics,optimizer=optimizer_manifest(model,optimizer)))
        del model,optimizer,criterion,logits,emb,aux,loss,incoming_before
        torch.cuda.empty_cache()
    return dict(passed=True,initialization=initial,initial_output_max_errors=errors,
                common_initialization_and_rng_matched=True,smoke=rows)


def prepare():
    cfg,dd,data,graphs,hashes,original=base_setup()
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json');verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['training_seed']==0 and plan['modes']==list(MODES) and plan['data_sha256']==hashes
        for name,digest in read(OUT/'artifacts_manifest.json').items():assert sha(OUT/name)==digest
        return cfg,dd,data,graphs,plan
    check_log=ROOT/'branch_fusion_checks.log'
    assert 'OK' in check_log.read_text(encoding='utf-8') and 'FAILED' not in check_log.read_text(encoding='utf-8')
    manifest,reference_audit=verify_reference(cfg,dd,data,graphs)
    checks=smoke(cfg,dd,data,graphs)
    ref_interventions=[]
    SET_Random(0);model=make_model(dd,cfg,graphs,'C').eval()
    for fold in FOLDS:
        folder=REFERENCE/f'seed_0/content/fold_{fold:02d}'
        cp=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
        model.load_state_dict(cp['state_dict'],strict=True)
        test=data['Mask'][fold-1][1]
        with torch.no_grad():logits=model(data['Feature'])[0]
        with np.load(folder/'predictions.npz') as saved:
            np.testing.assert_allclose(logits[test].cpu().numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
        result,diagnostic=interventions(model,data,test)
        assert abs(result['both']['metrics']['acc']-read(folder/'complete.json')['best']['acc'])<1e-12
        ref_interventions.append(dict(mode='C',seed=0,fold=fold,stage='best',
                                     interventions=result,diagnostics=diagnostic))
    del model,cp,logits
    torch.cuda.empty_cache()
    old_plan=read(ROOT/'Result/alternating_cooperative_seed0_20260923/PLAN.json')
    verify_manifest(old_plan['code_sha256'])
    sources=tuple(dict.fromkeys((*old_plan['code_sha256'],*NEW_SOURCES)))
    plan=dict(experiment='C branch fusion and capacity control',training_seed=0,split_seed=0,
        modes=list(MODES),folds=list(FOLDS),epochs=1000,planned_runs=20,reused_C_runs=10,
        class_names=dd['Class_Names'],modalities=dd['Modal_Name'],samples=dd['Sample_Num'],
        original_config=original['original_config'],class_weights=dd['Label_Weight'].cpu().tolist(),
        data_sha256=hashes,candidate_hash=original['candidate_hash'],transductive=True,
        checkpoint_selection=original['checkpoint_selection'],parameters=PARAMETERS,
        architecture=dict(C='96-d semantic+own -> 96/48/3 classifier',
            concat='Concat semantic and own -> 192/48/3 classifier',
            wide='Keep semantic+own -> 96/94/3 classifier; 8 fewer parameters than concat'),
        initialization='Fresh C seed0 per fold. Concat W=[W_C,W_C], other head weights copied. Wide first48 units copy C, extra inputs random and output weights zero. No trained-checkpoint warm start.',
        randomness='Original CPU/CUDA construction and backbone streams preserved. Wide first48 dropout units use original stream; extra46 use private Bernoulli generator seed8101, same p=.64. Private stream restarts each fresh fold; no mid-fold continuation.',
        optimizer=old_plan['optimizer'],objective=old_plan['objective'],
        run_order='For each original fold, concat then wide; 20 total new full fits',
        interventions='At fixed best/last checkpoints zero semantic or own immediately before fusion; no retraining/selection. C best only because no last checkpoint exists.',
        primary='Same seed/fold best ACC: concat-C and concat-wide, both required; capacity control is nearly matched, not identical function class.',
        secondary='Macro F1, raw-logit AUC, class recalls, fixed epoch1000, last100 mean and paired corrected/harmed cases.',
        scope='No new graph, new auxiliary loss, early stop, additional seeds, hyperparameter tuning or automatic follow-up.',
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
    dump(OUT/'reference_interventions.json',ref_interventions)
    dump(OUT/'PLAN.json',plan)
    dump(OUT/'artifacts_manifest.json',{name:sha(OUT/name) for name in
         ('data_splits_and_graphs.npz','implementation_checks.log','reference_interventions.json')})
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
                rows.append(dict(mode=mode,seed=0,fold=fold,stage=stage,max_abs_replay_error=error,
                                 interventions=result,diagnostics=diagnostic))
            diagnostic=read(folder/'diagnostics.json')
            assert sorted(map(int,diagnostic))==list(DIAGNOSTIC_EPOCHS)
            assert all(np.isfinite(list(v.values())).all() for v in diagnostic.values())
            assert sum(p.numel() for p in model.parameters())==PARAMETERS[mode]
            assert optimizer_manifest(model,relation_optimizer(model,cfg,0.))==read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result=dict(passed=True,verified_runs=20,verified_checkpoints=40,records=rows,
        reference_interventions=read(OUT/'reference_interventions.json'),
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

