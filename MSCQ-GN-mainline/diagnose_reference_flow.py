"""Inspect where off-diagonal graph messages weaken in fixed checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relation_experiment import ROOT, setup, dump, timestamp
from reference_selection_experiment import read
from Model.reference_selection import ReferenceSelectionModel, composed_self_part
from Model.relation_sheaf import cooperative_operator, conformal_maps


@torch.no_grad()
def flow(model,x,rows,labels=None):
    cache={}
    def collect(module,inputs): cache['tokens']=inputs[0].detach()
    handle=model.propagation.register_forward_pre_hook(collect)
    try: base_logits=model(x)[0]
    finally: handle.remove()
    p=model.propagation
    tokens=cache['tokens']; count,m,h=tokens.shape
    z=p.input_norm(tokens)
    q=F.normalize(p.edge_query(z),dim=-1).permute(1,0,2)
    q=q[None].expand(m,-1,-1,-1).reshape(m*m*count,-1)
    i,j=p.edge
    w=torch.sigmoid((q[i]*q[j]).sum(-1)+p.similarity*p.edge_similarity.flatten()[p.edge_group]
                    +p.edge_bias.flatten()[p.edge_group])
    maps=conformal_maps(p.map_values(z).reshape(-1,4))
    v=p.value(z).permute(1,0,2)[None].expand(m,-1,-1,-1).reshape(-1,2,h//2)
    full=cooperative_operator(v,p.edge,w,*maps)
    own=composed_self_part(v,p.edge,w,*maps,constant_maps=True)
    sources=(full-own).reshape(m,m,count,h)[:,:,rows]
    norms=sources.norm(dim=-1).mean(0)
    averaged=sources.mean(0)
    ratio=averaged.norm(dim=-1)/norms.clamp_min(1e-12)
    modal=[]
    for n in range(m):
        active=norms[n]>1e-5
        modal.append(dict(mean_individual_source_norm=float(norms[n].mean()),
            mean_after_averaging_norm=float(averaged[n].norm(dim=-1).mean()),
            cancellation_ratio_mean=float(ratio[n][active].mean()) if bool(active.any()) else None,
            active_rows=int(active.sum())))
    interventions=[]
    if labels is not None:
        rng=np.random.default_rng(44019)
        variants=[('cohort_mean',None)]+[(f'permutation_{k}',rng.permutation(count)) for k in range(3)]
        correct=base_logits[rows].argmax(-1)==labels[rows]
        def restore_operator(delta):
            delta=delta.reshape(m,m,count,h).mean(0).permute(1,0,2)
            return p.output_norm(tokens-p.step_logit.sigmoid()*p.output(F.gelu(delta)))
        torch.testing.assert_close(restore_operator(full),p(tokens),rtol=2e-5,atol=2e-5)
        values=v.reshape(m,m,count,2,h//2)
        for name,permutation in variants:
            changed=(values.mean(2,keepdim=True).expand_as(values) if permutation is None else
                     values[:,:,torch.as_tensor(permutation,device=x.device)])
            changed=changed.reshape_as(v)
            changed_full=cooperative_operator(changed,p.edge,w,*maps)
            changed_own=composed_self_part(changed,p.edge,w,*maps,constant_maps=True)
            replacement=restore_operator(own+changed_full-changed_own)
            handle=p.register_forward_hook(lambda module,inputs,output: replacement)
            try: new_logits=model(x)[0]
            finally: handle.remove()
            new_correct=new_logits[rows].argmax(-1)==labels[rows]
            interventions.append(dict(condition=name,base_acc=float(correct.double().mean()),
                new_acc=float(new_correct.double().mean()),
                delta_acc=float(new_correct.double().mean()-correct.double().mean()),
                changed_predictions=int((new_logits[rows].argmax(-1)!=base_logits[rows].argmax(-1)).sum()),
                nll_delta=float(F.cross_entropy(new_logits[rows],labels[rows])-F.cross_entropy(base_logits[rows],labels[rows]))))
    return dict(by_content_modality=modal,foreign_content_interventions=interventions,
        meaning='Ratio ||mean(messages)|| / mean(||message||); near one means little vector cancellation, not proof of predictive usefulness')


def main():
    out=ROOT/'Result/reference_signal_diagnosis'
    summary=read(out/'COMPLETE.json')
    cfg,dd,data,graphs,_=setup()
    results=[]
    with torch.no_grad():
        for entry in summary['records']:
            folder=out/f"inner_{entry['inner']}"/entry['policy']
            model=ReferenceSelectionModel(dd,cfg,graphs,entry['teacher_seed'],learned=False).to(cfg.Device).eval()
            model.load_state_dict(torch.load(folder/'last.pt',map_location=cfg.Device,weights_only=False)['state_dict'])
            with np.load(folder/f"coalitions_epoch_{entry['epochs']:04d}.npz") as saved:
                held=torch.as_tensor(saved['held_mask'],device=cfg.Device)
            results.append(dict(kind='teacher',inner=entry['inner'],policy=entry['policy'],
                                **flow(model,data['Feature'],held)))
        for seed in (0,1,2):
            for fold in range(1,11):
                folder=ROOT/f'Result/relation_mechanism_nodecay/seed_{seed}/content/fold_{fold:02d}'
                model=ReferenceSelectionModel(dd,cfg,graphs,seed,learned=False).to(cfg.Device).eval()
                checkpoint=torch.load(folder/'best.pt',map_location=cfg.Device,weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
                with np.load(folder/'predictions.npz') as saved:
                    np.testing.assert_allclose(model(data['Feature'])[0][data['Mask'][fold-1][1]].cpu().numpy(),
                                               saved['logits'],atol=2e-5,rtol=2e-5)
                results.append(dict(kind='selected_C',seed=seed,fold=fold,epoch=checkpoint['best']['epoch'],
                                    **flow(model,data['Feature'],data['Mask'][fold-1][1],data['Label'])))
            print(json.dumps(dict(event='fixed_C_content_audit',seed=seed)),flush=True)
    result=dict(analyzed_at=timestamp(),modalities=dd['Modal_Name'],records=results)
    dump(out/'relation_average_audit.json',result)
    for condition in ('cohort_mean','permutation_0','permutation_1','permutation_2'):
        rows=[v for r in results if r['kind']=='selected_C' for v in r['foreign_content_interventions'] if v['condition']==condition]
        print(json.dumps(dict(condition=condition,mean_delta_acc=float(np.mean([r['delta_acc'] for r in rows])),
                              mean_nll_delta=float(np.mean([r['nll_delta'] for r in rows])),
                              changed_predictions=sum(r['changed_predictions'] for r in rows))),flush=True)


if __name__=='__main__':
    main()
