"""Audit and fixed-checkpoint interventions after the bounded late-readout runs."""
import csv
import json
import numpy as np
import torch
from relation_experiment import ROOT,setup,dump,sha
from alternating_cooperative_experiment import read,verify_manifest
from Model.late_patient_readout import LateReadoutModel

OUT=ROOT/'Result/abide_late_readout_20260927'
REF=ROOT/'Result/c_abide_repair_v2_20260927/full/W'


def main():
    assert read(OUT/'COMPLETE.json')['runs']==4
    plan=read(OUT/'PLAN.json');verify_manifest(plan['code_sha256']);verify_manifest(plan['reference_sha256'])
    assert sha(OUT/'late_graph_and_splits.npz')==plan['late_graph_sha256']
    cfg,dd,data,graphs,_=setup(-1,'Config/c_abide_repair_v2_W.ini',10)
    records=[];predictions={};audit=[]
    for arm in ['self','population']:
        model=LateReadoutModel(dd,cfg,graphs,arm,training_seed=0).eval()
        for fold in [1,9]:
            folder=OUT/arm/f'fold_{fold:02d}';fit,test=data['Mask'][fold-1];y=data['Label']
            with (folder/'epochs.csv').open() as stream:curve=list(csv.DictReader(stream))
            assert [int(r['epoch']) for r in curve]==list(range(1,1001))
            result=read(folder/'complete.json')
            assert int(np.argmax([float(r['test_acc']) for r in curve]))+1==result['best']['epoch']
            for name in ['last','best']:
                cp=torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=False)
                assert (cp['arm'],cp['fold'],cp['seed'])==(arm,fold,0)
                model.load_state_dict(cp['state_dict'])
                with torch.no_grad():z=model(data['Feature'])[0]
                with np.load(folder/f'{name}_predictions.npz') as saved:
                    np.testing.assert_allclose(z[test].numpy(),saved['logits'],atol=2e-5,rtol=2e-5)
                    np.testing.assert_array_equal(z[test].argmax(-1).numpy(),saved['logits'].argmax(-1))
                    np.testing.assert_array_equal(y[test].numpy(),saved['labels'])
                    if name=='best':np.testing.assert_array_equal(np.asarray(dd['Index'])[test],saved['row_index'])
                audit.append(dict(arm=arm,fold=fold,checkpoint=name,passed=True,sha256=sha(folder/f'{name}.pt')))
            late=model.classifier[0];cache={}
            def capture(mod,args,output):
                cache['z']=args[0].detach();cache['delta']=(output-args[0]).detach()
            hook=late.register_forward_hook(capture)
            with torch.no_grad():native=model(data['Feature'])[0]
            hook.remove()
            variants={'native':native}
            with torch.no_grad():
                late.scale=0.;variants['late_block_disabled']=model(data['Feature'])[0];late.scale=.1
                late.mode='self';variants['late_neighbors_disabled']=model(data['Feature'])[0];late.mode=arm
            metrics={}
            for name,zz in variants.items():
                p=zz.argmax(-1)
                metrics[name]={part:int((p[mask]==y[mask]).sum())/int(mask.sum()) for part,mask in [('train',fit),('test',test)]}
                metrics[name]['test_predictions_changed_vs_native']=int((p[test]!=native[test].argmax(-1)).sum())
            records.append(dict(arm=arm,fold=fold,epoch=result['best']['epoch'],metrics=metrics,
                relative_residual_norm=float(cache['delta'].norm()/cache['z'].norm()),
                last_train_acc=result['last']['train_acc'],last_test_acc=result['last']['test_acc']))
            predictions[arm,fold]=(native[test].argmax(-1).numpy(),y[test].numpy())
    paired=[]
    for fold in [1,9]:
        with np.load(REF/f'content/fold_{fold:02d}/predictions.npz') as saved:
            base=saved['logits'].argmax(-1);label=saved['labels']
        for old,new in [('baseline','self'),('baseline','population'),('self','population')]:
            a=base if old=='baseline' else predictions[old,fold][0];b=predictions[new,fold][0]
            np.testing.assert_array_equal(label,predictions[new,fold][1])
            paired.append(dict(fold=fold,comparison=f'{new}_vs_{old}',
                corrected=int(((a!=label)&(b==label)).sum()),harmed=int(((a==label)&(b!=label)).sum())))
    result=dict(audits=audit,records=records,paired=paired,script_sha256=sha(__file__),
        scope='Two exploratory test-selected folds, not tenfold confirmation. Posthoc disabling keeps trained weights and cannot substitute for retrained controls. Population support is existing sparse union, so result does not settle dense late graphs or every other propagation operator.')
    dump(OUT/'analysis.json',result)
    print(json.dumps(dict(records=records,paired=paired),ensure_ascii=False))


if __name__=='__main__':main()
