"""Run only frozen A1 training seeds 1 and 2 on the original seed-0 splits.

Historical source/results stay read-only. No new model, tuning, or other arms.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
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
    assert_rng_equal, metrics, verify_manifest, save_predictions)
from Model.alternating_cooperative import AlternatingCooperativeModel
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from analyze_alternating_a1_multiseed import summarize, write_report, read_record

OUT = ROOT/'Result/alternating_a1_seeds12_20260924'
REFERENCE = ROOT/'Result/relation_mechanism_nodecay'
OLD = ROOT/'Result/alternating_cooperative_seed0_20260923'
C_AUDIT = ROOT/'Result/c_result_reaudit_20260924T015759Z/audit.json'
SEEDS = (1,2)
FOLDS = tuple(range(1,11))
DIAGNOSTIC_EPOCHS = (1,100,500,1000)
NEW_SOURCES = ('alternating_a1_seeds12_experiment.py','analyze_alternating_a1_multiseed.py')


def make_model(dd,cfg,graphs,seed):
    return AlternatingCooperativeModel(dd,cfg,graphs,'A1',training_seed=seed).to(cfg.Device)


def verify_sources(plan):
    verify_manifest(plan['code_sha256'])


def references(dd,data,graphs,old_plan):
    audit = read(C_AUDIT)
    assert audit['passed'] and audit['checkpoint_replays']['distinct_best_checkpoints']==30
    verify_manifest(audit['source_and_result_sha256'])
    manifest = dict(audit['source_and_result_sha256'])
    def freeze(path):
        manifest[str(path.relative_to(ROOT))] = sha(path)
    freeze(C_AUDIT)
    verify_manifest(old_plan['code_sha256'])
    verify_manifest(old_plan['reference_manifest'])
    for filename in ('PLAN.json','COMPLETE.json','audit.json','data_splits_and_graphs.npz'):
        freeze(OLD/filename)
    assert read(OLD/'audit.json')['verified_checkpoints']==80
    assert read(OLD/'audit.json')['passed']
    for seed in (0,1,2):
        protocol=read(REFERENCE/f'seed_{seed}/PROTOCOL.json')
        assert protocol['training_seed']==seed and protocol['split_seed']==0
        assert protocol['original_config']==old_plan['original_config']
        with np.load(REFERENCE/f'seed_{seed}/splits_and_graphs.npz') as saved:
            np.testing.assert_array_equal(saved['source_row'],dd['Index'])
            np.testing.assert_array_equal(saved['labels'],data['Label'].cpu().numpy())
            np.testing.assert_array_equal(saved['test_masks'],np.stack([t.cpu().numpy() for _,t in data['Mask']]))
            for i,graph in enumerate(graphs):
                np.testing.assert_array_equal(saved[f'edge_{i}'],graph['edge'].numpy())
                np.testing.assert_array_equal(saved[f'neighbors_{i}'],graph['neighbors'].numpy())
        for fold in FOLDS:
            read_record(REFERENCE/f'seed_{seed}/content/fold_{fold:02d}',seed,fold,'content')
    for fold in FOLDS:
        folder=OLD/f'seed_0/A1/fold_{fold:02d}'
        record=read_record(folder,0,fold,'A1')['record']
        for stage in ('best','last'):
            assert sha(folder/f'{stage}.pt')==record[f'{stage}_checkpoint_sha256']
        for filename in ('complete.json','best.json','epochs.csv','best.pt','last.pt',
                         'predictions.npz','last_predictions.npz','optimizer_groups.json','diagnostics.json'):
            freeze(folder/filename)
    return manifest


def initialization_and_smoke(cfg,dd,data,graphs,old_plan):
    initialization={}
    for seed in (0,1,2):
        SET_Random(seed)
        c=RelationSheafModel(dd,cfg,graphs,'content',training_seed=seed).to(cfg.Device)
        c_hash,c_rng=state_hash(c),rng_state()
        SET_Random(seed)
        a=make_model(dd,cfg,graphs,seed)
        a_hash,a_rng=state_hash(a),rng_state()
        assert c_hash==a_hash
        assert_rng_equal(c_rng,a_rng)
        assert sum(p.numel() for p in a.parameters())==895096
        initialization[str(seed)]=dict(state_sha256=a_hash,matched_C_initialization_and_rng=True,parameters=895096)
        groups=optimizer_manifest(a,relation_optimizer(a,cfg,0.))
        assert groups==read(REFERENCE/f'seed_{seed}/content/fold_01/optimizer_groups.json')
        if seed==0:
            assert a_hash==old_plan['smoke']['initial_model_sha256']
        # P contains no stochastic operations, so moving its second invocation
        # must preserve the original C training-forward random stream.
        streams=[]
        for model in (c,a):
            SET_Random(811+seed)
            with torch.no_grad():
                model.train()
                assert torch.isfinite(model(data['Feature'])[0]).all()
            streams.append(rng_state())
        assert_rng_equal(*streams)
        del c,a,model
        torch.cuda.empty_cache()
    assert len({v['state_sha256'] for v in initialization.values()})==3
    smoke=[]
    for seed in SEEDS:
        SET_Random(seed)
        model=make_model(dd,cfg,graphs,seed)
        assert state_hash(model)==initialization[str(seed)]['state_sha256']
        optimizer=relation_optimizer(model,cfg,0.)
        criterion=MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
        fit=data['Mask'][0][0]
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for step in range(8):
            model.capture_diagnostics=False
            model.train();optimizer.zero_grad(set_to_none=True)
            output=model(data['Feature'])
            loss=criterion(output[0],data['Label'],fit,output[1],output[2])
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),name
                if name.startswith('propagation.'):
                    assert p.grad is not None,name
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            model.eval();model.capture_diagnostics=step==7
            with torch.no_grad():
                assert torch.isfinite(model(data['Feature'])[0]).all()
        torch.cuda.synchronize()
        trace=model.last_diagnostics
        assert not trace['reused_same_weight_tensor'] and trace['mean_absolute_edge_weight_change']>0
        smoke.append(dict(seed=seed,passed=True,test_metrics_inspected=False,
            seconds_per_epoch=(time.perf_counter()-start)/8,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
        del model,optimizer,criterion,output,loss
        torch.cuda.empty_cache()
    return initialization,smoke


def prepare():
    cfg,dd,data,graphs,hashes,original=base_setup()
    old_plan=read(OLD/'PLAN.json')
    verify_manifest(old_plan['code_sha256'])
    if (OUT/'PLAN.json').exists():
        plan=read(OUT/'PLAN.json')
        verify_sources(plan);verify_manifest(plan['reference_manifest'])
        assert plan['data_sha256']==hashes and plan['training_seeds']==[1,2]
        assert plan['candidate_hash']==original['candidate_hash'] and plan['planned_runs']==20
        assert plan['frozen_seed0_plan_sha256']==sha(OLD/'PLAN.json')
        return cfg,dd,data,graphs,plan
    manifest=references(dd,data,graphs,old_plan)
    initialization,smoke=initialization_and_smoke(cfg,dd,data,graphs,old_plan)
    sources=tuple(dict.fromkeys((*old_plan['code_sha256'],*NEW_SOURCES)))
    plan=dict(experiment='Frozen A1 additional training seeds 1 and 2',training_seeds=[1,2],
        comparison_seeds=[0,1,2],split_seed=0,folds=list(FOLDS),epochs=1000,planned_runs=20,
        reused_a1_seed0_runs=10,reused_C_runs=30,mode='A1',parameters=895096,
        original_config=old_plan['original_config'],class_names=dd['Class_Names'],modalities=dd['Modal_Name'],
        samples=dd['Sample_Num'],class_weights=dd['Label_Weight'].cpu().tolist(),
        checkpoint_selection=old_plan['checkpoint_selection'],
        architecture=old_plan['architecture'],schedule=old_plan['schedules']['A1'],
        optimizer=old_plan['optimizer'],objective=old_plan['objective'],
        candidate_hash=old_plan['candidate_hash'],data_sha256=hashes,
        diagnostics=old_plan['diagnostics'],transductive=True,
        frozen_seed0_plan_sha256=sha(OLD/'PLAN.json'),
        primary='Mean of ten-fold first-best TEST ACC within each seed; equal mean over seeds 0/1/2. Paired A1-C for every same seed/fold.',
        secondary='Macro F1, raw-logit AUC, class recalls, fixed epoch 1000 and last-100 mean; no alternative checkpoint selection',
        training='Fresh model per fold; SET_Random(training_seed) and constructor training_seed both change. Config/split seed stays 0. No checkpoint warm start.',
        run_order='All ten folds of training seed 1, then all ten folds of training seed 2',
        scope='Only A1 seeds 1/2. Do not train C/N0/N1/A0 or rerun seed0. No tuning or automatic follow-up.',
        aggregation_caveat='Repeated seeds reuse the same patients; row-seed counts are descriptive, not independent new patients.',
        reference_manifest=manifest,code_sha256={name:sha(ROOT/name) for name in sources},
        initialization=initialization,smoke=smoke,
        reference_audit=dict(C_reused_audit=str(C_AUDIT.relative_to(ROOT)),C_best_checkpoint_replays=30,
            A1_seed0_reused_audit=str((OLD/'audit.json').relative_to(ROOT)),
            note='Previously completed inference audits reused after fresh hash checks; C has no last.pt.'),
        environment=dict(executable=sys.executable,python=sys.version,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'PLAN.json',plan)
    for source in sources:
        destination=OUT/'source_snapshot'/source
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/source).read_bytes())
    (OUT/'data_splits_and_graphs.npz').write_bytes((OLD/'data_splits_and_graphs.npz').read_bytes())
    dump(OUT/'artifacts_manifest.json',{'data_splits_and_graphs.npz':sha(OUT/'data_splits_and_graphs.npz')})
    summarize(OUT,plan)
    dump(OUT/'status.json',dict(state='prepared',updated_at=timestamp(),planned_runs=20))
    return cfg,dd,data,graphs,plan


def save_checkpoint(folder,stage,model,epoch,score,fold,seed,plan):
    temp=folder/f'{stage}.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,metrics=score,seed=seed,fold=fold,
                    mode='A1',protocol_sha256=sha(OUT/'PLAN.json')),temp)
    temp.replace(folder/f'{stage}.pt')


def train_arm(cfg, dd, data, graphs, plan, fold, seed):
    mode = 'A1'
    folder = OUT/f'seed_{seed}/{mode}/fold_{fold:02d}'
    if (folder/'complete.json').exists():
        return
    verify_sources(plan)
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(seed)
    model = make_model(dd, cfg, graphs, seed)
    initial_hash = state_hash(model)
    assert initial_hash == plan['initialization'][str(seed)]['state_sha256']
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
                save_checkpoint(folder, 'best', model, epoch, best, fold, seed, plan)
                save_predictions(folder/'predictions.npz', logits, y, test, dd['Index'])
                dump(folder/'best.json', best)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=mode,
                              fold=fold, seed=seed, epochs=1000, planned_runs=20,
                              parameters=parameters, best_acc=best['acc'], **row)
                dump(OUT/'status.json', status)
                print(json.dumps(status), flush=True)
            if epoch == 1000:
                last = dict(epoch=epoch, **metrics(logits[test], y[test]))
                save_checkpoint(folder, 'last', model, epoch, last, fold, seed, plan)
                save_predictions(folder/'last_predictions.npz', logits, y, test, dd['Index'])
    record = dict(mode=mode, fold=fold, seed=seed, parameters=parameters, best=best, last=last,
                  initial_model_sha256=initial_hash, mean_last100_acc=float(np.mean(history[-100:])),
                  best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
                  completed_at=timestamp(), seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    torch.cuda.empty_cache()
    summarize(OUT, plan)


def audit_completed(cfg, dd, data, graphs, plan):
    verify_sources(plan); verify_manifest(plan['reference_manifest'])
    for name, digest in read(OUT/'artifacts_manifest.json').items():
        assert sha(OUT/name) == digest
    rows = []
    for seed in SEEDS:
        mode = 'A1'
        SET_Random(seed)
        model = make_model(dd, cfg, graphs, seed).eval()
        for fold in FOLDS:
            folder = OUT/f'seed_{seed}/{mode}/fold_{fold:02d}'
            record = read(folder/'complete.json')
            assert (record['mode'], record['seed'], record['fold']) == (mode, seed, fold)
            assert record['initial_model_sha256'] == plan['initialization'][str(seed)]['state_sha256']
            with (folder/'epochs.csv').open(encoding='utf-8') as stream:
                epochs = list(csv.DictReader(stream))
            with (REFERENCE/f'seed_{seed}/content/fold_{fold:02d}/epochs.csv').open(encoding='utf-8') as stream:
                old_epochs = list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in epochs] == list(range(1,1001))
            assert np.isfinite([[float(r[k]) for k in ('train_loss','test_loss','test_acc','lr')] for r in epochs]).all()
            assert int(np.argmax([float(r['test_acc']) for r in epochs]))+1 == record['best']['epoch']
            np.testing.assert_array_equal([float(r['lr']) for r in epochs], [float(r['lr']) for r in old_epochs])
            test = data['Mask'][fold-1][1]
            for stage, filename in (('best','predictions.npz'),('last','last_predictions.npz')):
                assert sha(folder/f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                assert cp['protocol_sha256'] == sha(OUT/'PLAN.json')
                assert (cp['mode'],cp['seed'],cp['fold']) == (mode,seed,fold)
                assert cp['epoch'] == record[stage]['epoch']
                assert cp['metrics'] == record[stage]
                model.load_state_dict(cp['state_dict'], strict=True)
                model.capture_diagnostics = True
                with torch.no_grad():
                    logits = model(data['Feature'])[0]
                with np.load(folder/filename) as saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                    np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                    np.testing.assert_allclose(saved['logits'],logits[test].cpu().numpy(),atol=2e-5,rtol=2e-5)
                    error = float(np.max(np.abs(saved['logits']-logits[test].cpu().numpy())))
                actual = metrics(logits[test],data['Label'][test])
                for key in ('acc','auc','f1','macro_f1','nll','brier'):
                    assert abs(actual[key]-record[stage][key]) < 2e-5
                np.testing.assert_array_equal(actual['confusion_matrix'], record[stage]['confusion_matrix'])
                np.testing.assert_allclose(actual['per_class_recall'], record[stage]['per_class_recall'], atol=1e-12)
                diagnostic = model.last_diagnostics
                assert diagnostic['reused_same_weight_tensor'] == mode.endswith('0')
                if mode.endswith('0'):
                    assert diagnostic['mean_absolute_edge_weight_change'] == 0.
                rows.append(dict(mode=mode,seed=seed,fold=fold,stage=stage,max_abs_replay_error=error,diagnostics=diagnostic))
            diagnostics = read(folder/'diagnostics.json')
            assert sorted(map(int, diagnostics)) == list(DIAGNOSTIC_EPOCHS)
            assert sum(p.numel() for p in model.parameters()) == record['parameters'] == 895096
            for value in diagnostics.values():
                assert value['reused_same_weight_tensor'] == mode.endswith('0')
                assert np.isfinite(value['mean_absolute_edge_weight_change'])
                for call in ('first','second'):
                    assert np.isfinite(value[call]['graph_update_over_input_norm'])
            optimizer = relation_optimizer(model, cfg, 0.)
            assert optimizer_manifest(model, optimizer) == read(folder/'optimizer_groups.json')
        del model
        torch.cuda.empty_cache()
    result = dict(passed=True, verified_runs=20, verified_checkpoints=40, records=rows,
                  same_original_lr_schedule=True, same_original_source_and_reference_hashes=True)
    dump(OUT/'audit.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    cfg,dd,data,graphs,plan=prepare()
    if args.prepare:
        print(json.dumps(dict(state='prepared',planned_runs=20,initialization=plan['initialization'],smoke=plan['smoke'])),flush=True)
        return
    if (OUT/'COMPLETE.json').exists():
        print('Already complete; historical files retained.',flush=True)
        return
    lock=OUT/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for seed in SEEDS:
            for fold in FOLDS:
                train_arm(cfg,dd,data,graphs,plan,fold,seed)
        dump(OUT/'status.json',dict(state='auditing',pid=os.getpid(),updated_at=timestamp()))
        audit=audit_completed(cfg,dd,data,graphs,plan)
        summary=write_report(OUT,plan,audit)
        dump(OUT/'COMPLETE.json',dict(completed_at=timestamp(),audit=audit,summary=summary))
        dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated_at=timestamp(),completed_runs=20))
    except BaseException as error:
        dump(OUT/'status.json',dict(state='failed',pid=os.getpid(),updated_at=timestamp(),error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
