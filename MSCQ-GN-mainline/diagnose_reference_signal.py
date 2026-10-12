"""Locate weak teacher reference signals with matched internal-only controls.

No outer-test metrics, model selection, architecture change, or new mainline.
Two fixed inner partitions of seed 0 / outer fold 1; only reference-mask
training policy differs. All other training settings match the old teachers.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relation_experiment import ROOT, setup, dump, timestamp, sha
from reference_selection_experiment import read, model_hash, optim_and_loss, step
from Utils import SET_Random
from Utils.reference_selection import inner_splits, coalition_masks, mean_marginal_gain, training_class_weights
from Model.reference_selection import ReferenceSelectionModel
from Model.relation_sheaf import conformal_maps


def raw_signal_audit(source):
    absolute, signed = [], []
    for path in sorted(source.glob('seed_*/crossfit/fold_*/inner_*/coalitions.npz')):
        with np.load(path) as saved:
            losses = saved['losses']
            modalities = saved['masks'].shape[1]
            ids = np.arange(len(losses))
            marginals = np.stack([losses[ids[(ids & (1 << m)) == 0]] -
                losses[ids[(ids & (1 << m)) == 0] | (1 << m)] for m in range(modalities)])
            absolute.append(np.abs(marginals).mean(1).T)
            signed.append(np.abs(marginals.mean(1)).T)
    absolute, signed = np.concatenate(absolute), np.concatenate(signed)
    active = absolute > 1e-5
    return dict(mean_absolute_unaveraged_gain_quantiles=np.quantile(absolute, [.5,.9,.95,.99]).tolist(),
        absolute_mean_gain_quantiles=np.quantile(signed, [.5,.9,.95,.99]).tolist(),
        cancellation_ratio_quantiles_where_signal_above_1e5=np.quantile(signed[active]/absolute[active], [.1,.5,.9]).tolist(),
        fraction_active_with_more_than_half_cancelled=float((signed[active] < .5*absolute[active]).mean()),
        conclusion='Weak gains already precede averaging; cancellation is uncommon among nontrivial signals.')


@torch.no_grad()
def probe(model, x, y, held, fit, epoch, folder, exhaustive=False):
    model.eval()
    saved = {}
    def capture(name):
        def hook(module, inputs, output):
            saved[name] = output.detach().clone()
            if name == 'propagated':
                saved['tokens'] = inputs[0].detach().clone()
        return hook
    hooks = [model.propagation.register_forward_hook(capture('propagated')),
             model.Message_MLP.register_forward_hook(capture('semantic')),
             model.Global_Message.register_forward_hook(capture('global'))]
    m = model.propagation.modalities
    try:
        full = model(x, torch.ones(1, m, device=x.device))[0]
        opened = {k: v.clone() for k, v in saved.items()}
        neighbor_norm = model.propagation.last_neighbor_norm.clone()
        closed = model(x, torch.zeros(1, m, device=x.device))[0]
        shut = {k: v.clone() for k, v in saved.items()}
    finally:
        for hook in hooks:
            hook.remove()
    full_loss = F.cross_entropy(full[held], y[held], reduction='none')
    closed_loss = F.cross_entropy(closed[held], y[held], reduction='none')
    single_gains = []
    for n in range(m):
        mask = torch.ones(1, m, device=x.device); mask[:, n] = 0
        prediction = model(x, mask)[0]
        single_gains.append(F.cross_entropy(prediction[held], y[held], reduction='none')-full_loss)
    gain = torch.stack(single_gains, -1)
    z = model.propagation.input_norm(opened['tokens'])
    parameters = model.propagation.map_values(z)
    _, _, send, receive = conformal_maps(parameters)
    before = opened['tokens'][held]
    delta = opened['propagated'][held]-shut['propagated'][held]
    semantic_delta = opened['semantic'][held]-shut['semantic'][held]
    record = dict(epoch=epoch, held_count=int(held.sum()),
        fit_acc_full=float((full[fit].argmax(-1)==y[fit]).float().mean()),
        held_acc_full=float((full[held].argmax(-1)==y[held]).float().mean()),
        held_acc_closed=float((closed[held].argmax(-1)==y[held]).float().mean()),
        held_nll_full=float(full_loss.mean()), held_nll_closed=float(closed_loss.mean()),
        held_abs_nll_on_off_mean=float((closed_loss-full_loss).abs().mean()),
        held_abs_logit_on_off_mean=float((full[held]-closed[held]).abs().mean()),
        held_changed_predictions=int((full[held].argmax(-1)!=closed[held].argmax(-1)).sum()),
        input_patient_std_by_modality=before.std(0).norm(dim=-1).cpu().tolist(),
        neighbor_norm_by_modality=neighbor_norm[held].mean(0).cpu().tolist(),
        propagated_delta_norm_by_modality=delta.norm(dim=-1).mean(0).cpu().tolist(),
        propagated_delta_relative=float(delta.norm() / opened['propagated'][held].norm().clamp_min(1e-12)),
        semantic_delta_norm=float(semantic_delta.norm(dim=-1).mean()),
        semantic_norm=float(opened['semantic'][held].norm(dim=-1).mean()),
        global_norm=float(opened['global'][held].norm(dim=-1).mean()),
        send_scale_by_modality=send.mean((0,2)).cpu().tolist(),
        receive_scale_by_modality=receive.mean((0,2)).cpu().tolist(),
        graph_step=float(model.propagation.step_logit.sigmoid()),
        full_context_abs_gain_mean=float(gain.abs().mean()),
        full_context_abs_gain_p95=float(torch.quantile(gain.abs().flatten(), .95)),
        full_context_target_within_001_half=float(((gain/.05).sigmoid().sub(.5).abs()<.01).float().mean()))
    if exhaustive:
        masks = coalition_masks(m)
        losses = np.stack([F.cross_entropy(model(x, mask[None])[0][held], y[held], reduction='none').cpu().numpy()
                           for mask in masks])
        gains = mean_marginal_gain(losses, m)
        target = 1/(1+np.exp(-np.clip(gains/.05, -80, 80)))
        record['coalitions'] = dict(abs_gain_mean=float(np.abs(gains).mean()),
            abs_gain_p95=float(np.quantile(np.abs(gains), .95)),
            target_within_001_half=float((np.abs(target-.5)<.01).mean()),
            nll_range_mean=float(np.ptp(losses, axis=0).mean()),
            nll_range_below_001=float((np.ptp(losses, axis=0)<.001).mean()))
        np.savez_compressed(folder/f'coalitions_epoch_{epoch:04d}.npz', losses=losses, masks=masks,
                            gains=gains, held_mask=held.cpu().numpy())
    return record


def summarize(out, args):
    result = dict(updated_at=timestamp(), records=[],
        scope='One prespecified outer fold, two inner partitions; fixed teacher checkpoints, no outer test evaluation')
    for inner in args.inners:
        for policy in ('random_mask', 'always_open'):
            path = out/f'inner_{inner}'/policy/'complete.json'
            if path.exists():
                result['records'].append(read(path))
    if len(result['records']) == 2*len(args.inners):
        for inner in args.inners:
            a,b = [r for r in result['records'] if r['inner']==inner]
            assert a['initial_state_sha256'] == b['initial_state_sha256']
            assert a['fit_mask_sha256'] == b['fit_mask_sha256']
            assert a['teacher_seed'] == b['teacher_seed']
        result['paired_initialization_and_masks_verified'] = True
    dump(out/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='Result/reference_signal_diagnosis')
    parser.add_argument('--source', default='Result/reference_selection')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--inners', type=int, nargs='+', default=[0,1])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out, source = ROOT/args.output, ROOT/args.source
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    protocol = read(source/'PROTOCOL.json')
    for path,digest in protocol['code_sha256'].items():
        assert sha(ROOT/path)==digest, f'Old experiment source changed: {path}'
    cfg, dd, data, graphs, _ = setup(k=protocol['candidate_k'])
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(created_from_protocol_sha256=sha(source/'PROTOCOL.json'),
        diagnostic_source_sha256=sha(Path(__file__)), epochs=args.epochs, outer_fold=1, training_seed=0,
        inner_partitions=args.inners, policies=['random_mask','always_open'],
        probes=[0,1,25,50,100,200,400,600,800,1000], inner_split_seed=31010,
        labels='Only inner fitting labels in training; inner holdout labels for fixed diagnostic probes; no outer-test metrics',
        checkpoint_policy='fixed epochs, no checkpoint selected, no tuning',
        change='Only mask training policy; initial weights, random stream, supervised mask, optimizer and scheduler identical')
    if (out/'PLAN.json').exists() and read(out/'PLAN.json') != plan:
        raise ValueError('Diagnostic code or plan changed')
    dump(out/'PLAN.json', plan)
    dump(out/'saved_signal_audit.json', raw_signal_audit(source))
    x, y = data['Feature'], data['Label']
    outer_fit, outer_test = data['Mask'][0]
    partitions = inner_splits(y, outer_fit, outer_test, 31010, 2)
    probes = {0,1,25,50,100,200,400,600,800,args.epochs}
    try:
        for inner in args.inners:
            train_np, held_np = partitions[inner]
            train = torch.as_tensor(train_np, device=cfg.Device)
            held = torch.as_tensor(held_np, device=cfg.Device)
            # The same split must be the original teacher split.
            with np.load(source/f'seed_0/crossfit/fold_01/inner_{inner}/coalitions.npz') as saved:
                np.testing.assert_array_equal(saved['train_mask'], train_np)
                np.testing.assert_array_equal(saved['held_mask'], held_np)
            for policy in ('random_mask','always_open'):
                folder = out/f'inner_{inner}'/policy
                if (folder/'complete.json').exists():
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                teacher_seed = 52003+inner
                SET_Random(teacher_seed)
                model = ReferenceSelectionModel(dd,cfg,graphs,teacher_seed,learned=False).to(cfg.Device)
                initial_hash = model_hash(model)
                teacher_dd = dict(dd,Label_Weight=training_class_weights(y,train,dd['Class_Num']))
                criterion, optimizer, scheduler = optim_and_loss(model,cfg,teacher_dd,folder)
                teacher_y = torch.where(train,y,torch.zeros_like(y))
                generator = torch.Generator(device=cfg.Device).manual_seed(teacher_seed+9100)
                observations = [probe(model,x,y,held,train,0,folder,exhaustive=False)]
                started = time.perf_counter()
                with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream); writer.writerow(['epoch','train_loss','lr','seconds'])
                    for epoch in range(1,args.epochs+1):
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        group=torch.rand(len(y),1,generator=generator,device=cfg.Device)
                        mask=(torch.rand(len(y),len(dd['Modal_Name']),generator=generator,device=cfg.Device)<.5).float()
                        mask=torch.where(group<.25,torch.zeros_like(mask),mask)
                        mask=torch.where(group>=.75,torch.ones_like(mask),mask)
                        if policy=='always_open':
                            mask=torch.ones_like(mask)
                        logits,embeddings,auxiliary=model(x,mask)
                        loss=criterion(logits,teacher_y,train,embeddings,auxiliary)
                        step(model,optimizer,loss,cfg)
                        writer.writerow([epoch,float(loss.detach()),optimizer.param_groups[0]['lr'],time.perf_counter()-started])
                        if scheduler: scheduler.step()
                        if epoch in probes:
                            observation=probe(model,x,y,held,train,epoch,folder,exhaustive=epoch in (200,args.epochs))
                            observations.append(observation)
                            dump(folder/'trace.json', observations)
                            print(json.dumps(dict(event='probe',inner=inner,policy=policy,**observation)),flush=True)
                        if epoch==1 or epoch%25==0 or epoch==args.epochs:
                            stream.flush()
                            status=dict(state='running',updated_at=timestamp(),inner=inner,policy=policy,epoch=epoch,
                                        train_loss=float(loss.detach()),seconds=time.perf_counter()-started)
                            dump(out/'status.json',status)
                torch.save(dict(state_dict=model.state_dict(),teacher_seed=teacher_seed,epoch=args.epochs,
                                policy=policy,inner=inner),folder/'last.pt')
                import hashlib
                record=dict(inner=inner,policy=policy,teacher_seed=teacher_seed,epochs=args.epochs,
                    initial_state_sha256=initial_hash,fit_mask_sha256=hashlib.sha256(train_np.tobytes()).hexdigest(),
                    completed_at=timestamp(),seconds=time.perf_counter()-started,trace=observations)
                dump(folder/'complete.json',record)
                summarize(out,args)
                del model,optimizer,criterion,loss,logits,embeddings,auxiliary
        result=summarize(out,args)
        dump(out/'COMPLETE.json',result)
        dump(out/'status.json',dict(state='complete',updated_at=timestamp()))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',updated_at=timestamp(),error=repr(error)))
        raise


if __name__=='__main__':
    main()
