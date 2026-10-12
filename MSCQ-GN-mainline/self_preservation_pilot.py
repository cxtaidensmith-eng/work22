"""Budget-matched warm-start pilot: self-only, frozen-self graph, joint graph.

Fold 1 and seeds 0/1/2 are fixed before training. A fixed 200-epoch self-only
warm-up is shared; each arm then continues to epoch 1000. Test labels retain
their historical logging/first-maximum selection role, never choosing warm-up.
"""
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from relation_experiment import ROOT, setup, dump, sha, timestamp, scores
from reference_selection_experiment import read, model_hash, step
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import graph_only_parameter, optimizer_manifest
from Model.relation_sheaf import RelationSheafModel
from Loss import MSCQGNLoss

WARMUP, EPOCHS = 200, 1000
SEEDS = (0, 1, 2)
ARMS = ('self_only', 'frozen_self', 'joint')
PROBES = (200, 201, 250, 400, 600, 800, 1000)


def patient_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not graph_only_parameter(name):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def new_optimizer(model, cfg):
    groups = []
    for graph in (False, True):
        parameters = [p for name, p in model.named_parameters() if graph_only_parameter(name) == graph]
        if parameters:
            groups.append(dict(params=parameters, weight_decay=0. if graph else cfg.weight_decay))
    optimizer = torch.optim.Adam(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min) if cfg.Use_scheduler else None
    return optimizer, scheduler


def named_optimizer_state(model, optimizer):
    return {name: copy.deepcopy(optimizer.state[p]) for name, p in model.named_parameters() if p in optimizer.state}


def restore_optimizer(model, cfg, named_state, scheduler_state, lr):
    optimizer, scheduler = new_optimizer(model, cfg)
    for name, parameter in model.named_parameters():
        if name in named_state:
            optimizer.state[parameter] = copy.deepcopy(named_state[name])
            if 'step' in optimizer.state[parameter]:
                # Plain Adam keeps its scalar step on CPU, even when a saved
                # warm-up bundle was mapped to CUDA together with its moments.
                optimizer.state[parameter]['step'] = optimizer.state[parameter]['step'].cpu()
    if scheduler is not None:
        scheduler.load_state_dict(copy.deepcopy(scheduler_state))
        # The graph arm adds a parameter group; both follow the original clock.
        scheduler.base_lrs = [cfg.lr]*len(optimizer.param_groups)
        scheduler._last_lr = [lr]*len(optimizer.param_groups)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return optimizer, scheduler


def make_arm(dd, cfg, graphs, seed, arm, self_state):
    model = RelationSheafModel(dd, cfg, graphs, 'self_only' if arm == 'self_only' else 'content',
                              training_seed=seed).to(cfg.Device)
    if arm == 'self_only':
        model.load_state_dict(self_state)
    else:
        incompatible = model.load_state_dict(self_state, strict=False)
        assert not incompatible.unexpected_keys
        assert all(graph_only_parameter(name) for name in incompatible.missing_keys)
        # Add the existing C operator as an initially zero residual, preserving
        # the warm self predictor exactly in both graph arms.
        with torch.no_grad():
            model.propagation.output.weight.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(arm != 'frozen_self' or graph_only_parameter(name))
    return model


@torch.no_grad()
def bypass_logits(model, x):
    handle = model.propagation.register_forward_hook(
        lambda module, inputs, output: module.output_norm(inputs[0]))
    try:
        return model(x)[0]
    finally:
        handle.remove()


@torch.no_grad()
def diagnostic(model, x, y, test, warm_logits, epoch, expected_patient_hash, frozen):
    model.eval()
    full = model(x)[0]
    own = bypass_logits(model, x)
    if frozen:
        assert patient_hash(model) == expected_patient_hash
        torch.testing.assert_close(own, warm_logits, atol=2e-5, rtol=2e-5)
    base_correct = warm_logits[test].argmax(-1) == y[test]
    full_correct = full[test].argmax(-1) == y[test]
    own_correct = own[test].argmax(-1) == y[test]
    return dict(epoch=epoch, full_acc=float(full_correct.double().mean()),
        bypass_acc=float(own_correct.double().mean()),
        bypass_abs_logit_drift_from_warm=float((own[test]-warm_logits[test]).abs().mean()),
        bypass_changed_from_warm=int((own[test].argmax(-1)!=warm_logits[test].argmax(-1)).sum()),
        full_corrected_vs_warm=int((~base_correct & full_correct).sum()),
        full_harmed_vs_warm=int((base_correct & ~full_correct).sum()),
        bypass_corrected_vs_warm=int((~base_correct & own_correct).sum()),
        bypass_harmed_vs_warm=int((base_correct & ~own_correct).sum()),
        graph_corrected_vs_current_bypass=int((~own_correct & full_correct).sum()),
        graph_harmed_vs_current_bypass=int((own_correct & ~full_correct).sum())), full, own


def prediction_file(path, logits, y, test, dd):
    np.savez_compressed(path, logits=logits[test].detach().cpu().numpy(),
        labels=y[test].cpu().numpy(), row_index=np.asarray(dd['Index'])[test.cpu().numpy()])


def run_epochs(out, folder, model, optimizer, scheduler, criterion, cfg, dd, data, seed, arm,
               start, stop, history=None, inherited_best=None, warm_logits=None, expected_patient_hash=None):
    folder.mkdir(parents=True, exist_ok=True)
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][0]
    train_y = torch.where(fit, y, torch.zeros_like(y))
    history = copy.deepcopy(history or [])
    best = copy.deepcopy(inherited_best)
    best_correct = int(round(best['acc']*int(test.sum()))) if best else -1
    post_best, post_correct = None, -1
    diagnostics = []
    started = time.perf_counter()
    fields = ['epoch', 'train_loss', 'test_acc', 'test_nll', 'lr', 'seconds']
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        for epoch in range(start, stop+1):
            model.train()
            # Frozen arms retain the same noise/dropout policy. Only updates
            # of patient parameters are disabled; eval-mode is not substituted.
            optimizer.zero_grad(set_to_none=True)
            logits, embedding, auxiliary = model(x)
            loss = criterion(logits, train_y, fit, embedding, auxiliary)
            step(model, optimizer, loss, cfg)
            model.eval()
            with torch.no_grad():
                logits = model(x)[0]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite prediction')
                correct = int((logits[test].argmax(-1)==y[test]).sum())
                nll = float(F.cross_entropy(logits[test], y[test]))
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_acc=correct/int(test.sum()),
                test_nll=nll, lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            writer.writerow(row)
            history.append(row)
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, phase='warmup' if epoch<=WARMUP else 'continuation', **scores(logits[test], y[test]))
                torch.save(dict(state_dict=model.state_dict(), best=best, arm=arm, seed=seed), folder/'best.pt')
                prediction_file(folder/'predictions.npz', logits, y, test, dd)
            if epoch > WARMUP and correct > post_correct:
                post_correct = correct
                post_best = dict(epoch=epoch, **scores(logits[test], y[test]))
                prediction_file(folder/'post_best_predictions.npz', logits, y, test, dd)
            if warm_logits is not None and epoch in PROBES:
                record, full, own = diagnostic(model,x,y,test,warm_logits,epoch,expected_patient_hash,arm=='frozen_self')
                diagnostics.append(record)
                dump(folder/'diagnostics.json', diagnostics)
                np.savez_compressed(folder/f'probe_{epoch:04d}.npz', full=full[test].cpu().numpy(),
                    bypass=own[test].cpu().numpy(), labels=y[test].cpu().numpy(), source_row=np.asarray(dd['Index'])[test.cpu().numpy()])
            if scheduler:
                scheduler.step()
            if epoch == start or epoch % 100 == 0:
                stream.flush()
                status = dict(state='training', seed=seed, arm=arm, epoch=epoch,
                    acc=row['test_acc'], best_acc=best['acc'], updated_at=timestamp())
                dump(out/'status.json', status)
                print(json.dumps(status), flush=True)
    torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=seed, epoch=stop), folder/'last.pt')
    prediction_file(folder/'last_predictions.npz', logits, y, test, dd)
    record = dict(seed=seed, arm=arm, best=best, post_warmup_best=post_best,
        last=dict(epoch=stop, **scores(logits[test], y[test])),
        mean_last100_acc=float(np.mean([r['test_acc'] for r in history if r['epoch']>stop-100])),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        patient_state_sha256=patient_hash(model), seconds=time.perf_counter()-started,
        best_checkpoint_sha256=sha(folder/'best.pt'), last_checkpoint_sha256=sha(folder/'last.pt'),
        completed_at=timestamp())
    dump(folder/'complete.json', record)
    return history, record


def summarize(out):
    entries = []
    for seed in SEEDS:
        for arm in ARMS:
            path = out/f'seed_{seed}'/arm/'complete.json'
            if path.exists():
                entries.append(read(path))
    result = dict(updated_at=timestamp(), completed=len(entries), planned=9, records=entries,
        scope='Fixed outer fold 1, same 60 test patients at three seeds; not a ten-fold result',
        caveat='Historical first TEST-ACC maximum; diagnostic hypotheses only')
    dump(out/'summary.json', result)
    return result


def main():
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT/'Result/self_preservation_pilot'
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    original_path = ROOT/'Result/reference_selection/PROTOCOL.json'
    original = read(original_path)
    for path, digest in original['code_sha256'].items():
        assert sha(ROOT/path) == digest, path
    cfg, dd, data, graphs, paths = setup(k=original['candidate_k'])
    assert {Path(p).name:sha(p) for p in paths} == original['data_sha256']
    assert all(getattr(cfg,k)==v for k,v in original['original_config'].items())
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest() == original['candidate_hash']
    plan = dict(experiment='Self preservation warm-start pilot', folds=[1], seeds=list(SEEDS), arms=list(ARMS),
        total_epochs=EPOCHS, warmup_epochs=WARMUP, shared_warmup='Fixed last epoch 200, never TEST-selected initialization',
        continuation_epochs=EPOCHS-WARMUP, architecture='Original self-only or C; no new module, graph, loss, or label input',
        identity_start='Zero C propagation.output.weight in BOTH graph arms; every initial prediction matches warm self model',
        frozen_scope='All non-graph parameters incl output LayerNorm frozen; propagation.* except output_norm are trainable',
        dropout='Original train-mode input noise, dropout and drop path retained in every arm including frozen weights',
        optimization='Adam patient moments and scheduler clock carried from warm-up; graph moments initially empty',
        graph_weight_decay=0., patient_weight_decay=cfg.weight_decay,
        randomness='Same continuation RNG state for all three arms; graph arms identical starting state',
        clipping='Original global clip=1 retained; freezing changes which gradients contribute to that norm',
        primary_selection='First maximum TEST ACC across all 1000 epochs including shared warm-up; report phase explicitly',
        secondary_metrics='First maximum after warm-up, fixed epoch 1000, last100 mean; no change to primary selection',
        probes=list(PROBES), labels='Train-mask labels only in gradients; historical cohort class weights retained; test labels for original logging/selection',
        interpretation='Freeze versus joint includes optimization/regularization effects; an advantage alone does not prove gradient conflict',
        expansion='Finish all three seeds before interpretation; no automatic ten-fold expansion',
        runner_sha256=sha(Path(__file__)), source_protocol_sha256=sha(original_path))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        assert read(out/'PLAN.json') == plan, 'Frozen plan changed'
    else:
        dump(out/'PLAN.json', plan)
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    try:
        for seed in SEEDS:
            warm_dir = out/f'seed_{seed}'/'warmup'
            bundle_path = warm_dir/'continuation_state.pt'
            if not bundle_path.exists():
                SET_Random(seed)
                model = RelationSheafModel(dd,cfg,graphs,'self_only',training_seed=seed).to(cfg.Device)
                optimizer, scheduler = new_optimizer(model,cfg)
                history, record = run_epochs(out,warm_dir,model,optimizer,scheduler,criterion,cfg,dd,data,
                    seed,'warmup',1,WARMUP)
                model.eval()
                with torch.no_grad():
                    warm_logits = model(data['Feature'])[0]
                bundle = dict(state_dict=copy.deepcopy(model.state_dict()), optimizer=named_optimizer_state(model,optimizer),
                    scheduler=copy.deepcopy(scheduler.state_dict()) if scheduler else None,
                    lr=optimizer.param_groups[0]['lr'], rng=capture_rng(), history=history, record=record,
                    logits=warm_logits, patient_hash=patient_hash(model))
                torch.save(bundle,bundle_path)
                del model,optimizer,scheduler,warm_logits,bundle
            bundle = torch.load(bundle_path,map_location=cfg.Device,weights_only=False)
            graph_initial_hash = None
            for arm in ARMS:
                folder = out/f'seed_{seed}'/arm
                if (folder/'complete.json').exists():
                    continue
                folder.mkdir(parents=True,exist_ok=True)
                model = make_arm(dd,cfg,graphs,seed,arm,bundle['state_dict'])
                model.eval()
                assert patient_hash(model)==bundle['patient_hash']
                with torch.no_grad():
                    torch.testing.assert_close(model(data['Feature'])[0],bundle['logits'],atol=2e-5,rtol=2e-5)
                initial = model_hash(model)
                if arm != 'self_only':
                    if graph_initial_hash is None:
                        graph_initial_hash = initial
                    else:
                        assert graph_initial_hash == initial
                optimizer,scheduler = restore_optimizer(model,cfg,bundle['optimizer'],bundle['scheduler'],bundle['lr'])
                dump(folder/'optimizer_groups.json',optimizer_manifest(model,optimizer))
                dump(folder/'initialization.json',dict(model_sha256=initial,patient_sha256=patient_hash(model),
                    warmup_bundle_sha256=sha(bundle_path),initial_predictions_match=True,
                    warmup_epoch=WARMUP,continuation_initial_lr=bundle['lr'],scheduler_last_epoch=scheduler.last_epoch))
                for filename in ('best.pt','predictions.npz'):
                    shutil.copyfile(warm_dir/filename,folder/filename)
                restore_rng(bundle['rng'])
                _,record = run_epochs(out,folder,model,optimizer,scheduler,criterion,cfg,dd,data,seed,arm,WARMUP+1,EPOCHS,
                    bundle['history'],bundle['record']['best'],bundle['logits'],bundle['patient_hash'])
                if arm=='frozen_self':
                    assert patient_hash(model)==bundle['patient_hash']
                summarize(out)
                print(json.dumps(dict(event='arm_complete',seed=seed,arm=arm,best=record['best']['acc'],
                    post_best=record['post_warmup_best']['acc'],last=record['last']['acc'])),flush=True)
                del model,optimizer,scheduler
            del bundle
        summary = summarize(out)
        assert summary['completed']==9
        dump(out/'COMPLETE.json',summary)
        dump(out/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',error=repr(error),updated_at=timestamp()))
        raise


if __name__=='__main__':
    main()
