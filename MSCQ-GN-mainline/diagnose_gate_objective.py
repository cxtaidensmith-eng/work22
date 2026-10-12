"""Frozen-checkpoint gate objective audit; no optimization or selection."""
import json
import numpy as np
import torch
from relation_experiment import ROOT, setup, dump, timestamp
from reference_selection_experiment import read, folder_for
from Model.reference_selection import ReferenceSelectionModel
from Loss import MSCQGNLoss


def main():
    source = ROOT/'Result/reference_selection'
    out = ROOT/'Result/reference_signal_diagnosis'
    cfg,dd,data,graphs,_ = setup()
    criterion = MSCQGNLoss(dd,cfg.Device,rate=cfg.Loss_rate,aux_loss_weight=cfg.aux_loss_weight)
    protocol=read(source/'PROTOCOL.json')
    rows=[]
    for mode in ('gate','gain_gate'):
        for seed in protocol['seeds']:
            for fold in protocol['folds']:
                path=folder_for(source,seed,mode,fold)
                model=ReferenceSelectionModel(dd,cfg,graphs,seed).to(cfg.Device).eval()
                model.load_state_dict(torch.load(path/'best.pt',map_location=cfg.Device,weights_only=False)['state_dict'])
                fit,_=data['Mask'][fold-1]
                with np.load(folder_for(source,seed,'crossfit',fold)/'targets.npz') as saved:
                    gains=torch.as_tensor(saved['gains'][fit.cpu().numpy()],device=cfg.Device)
                logits,embeddings,auxiliary=model(data['Feature'])
                loss=criterion(logits,data['Label'],fit,embeddings,auxiliary)
                gate_logits=model.propagation.last_gate_logits
                main_grad=torch.autograd.grad(loss,gate_logits)[0][fit].detach()
                p=gate_logits[fit].detach().sigmoid()
                targets=(gains/protocol['gain_temperature']).sigmoid()
                extra_grad=protocol['gain_weight']*(p-targets)/p.numel()
                neutral_grad=protocol['gain_weight']*(p-.5)/p.numel()
                def cosine(a,b):
                    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
                rows.append(dict(mode=mode,seed=seed,fold=fold,
                    main_gradient_norm=float(main_grad.norm()),gain_gradient_norm=float(extra_grad.norm()),
                    weighted_gain_to_main_norm_ratio=(float(extra_grad.norm()/main_grad.norm())
                        if float(main_grad.norm()) > 1e-12 else None),
                    cosine_gain_with_neutral=cosine(extra_grad,neutral_grad),
                    cosine_gain_with_main=cosine(extra_grad,main_grad),
                    neutral_to_actual_gradient_difference_ratio=float((extra_grad-neutral_grad).norm()/extra_grad.norm().clamp_min(1e-30)),
                    fraction_targets_within_001_half=float(((targets-.5).abs()<.01).float().mean()),
                    fraction_hard_clamped_gates=float(((model.propagation.last_gate[fit]==0)|(model.propagation.last_gate[fit]==1)).float().mean())))
                del model,loss,logits,embeddings,auxiliary,gate_logits
            print(json.dumps(dict(stage='gradient_audit',mode=mode,seed=seed)),flush=True)
    aggregate={}
    for mode in ('gate','gain_gate'):
        group=[r for r in rows if r['mode']==mode]
        keys=['weighted_gain_to_main_norm_ratio','cosine_gain_with_neutral','cosine_gain_with_main',
              'neutral_to_actual_gradient_difference_ratio','fraction_hard_clamped_gates']
        aggregate[mode]={k:dict(mean=float(np.mean([r[k] for r in group if r[k] is not None])),
            median=float(np.median([r[k] for r in group if r[k] is not None])),
            undefined_count=sum(r[k] is None for r in group)) for k in keys}
    result=dict(created_at=timestamp(),summary=aggregate,rows=rows,
        scope='Gradients w.r.t. gate logits on original training masks, evaluation mode, selected frozen checkpoints; no training trajectory claim',
        ordinary_gate_note='Gain gradients for ordinary gate are counterfactual: this objective was not used in its training')
    dump(out/'gate_objective_audit.json',result)
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':
    main()
