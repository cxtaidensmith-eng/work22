"""Post-hoc readout audit of ten existing W best checkpoints, without fitting."""
import json
import numpy as np
import torch
from relation_experiment import ROOT,setup,dump,sha
from Model.relation_sheaf import RelationSheafModel

def main():
    torch.set_num_threads(1)
    cfg,dd,data,graphs,_=setup(-1,'Config/c_abide_repair_v2_W.ini',10)
    ref=ROOT/'Result/c_abide_repair_v2_20260927/full/W'
    model=RelationSheafModel(dd,cfg,graphs,'content',training_seed=0).eval()
    records=[]
    for fold in range(1,11):
        folder=ref/f'content/fold_{fold:02d}'
        cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(cp['state_dict']); fit,test=data['Mask'][fold-1]; y=data['Label']
        with torch.no_grad():
            main_z,_,aux=model(data['Feature'])
            with np.load(folder/'predictions.npz') as saved:
                np.testing.assert_array_equal(main_z[test].argmax(1).numpy(),saved['logits'].argmax(1))
            all_pred={'main':main_z.argmax(1),
                'aux_combined':torch.stack([v.softmax(-1)[:,1] for v in aux],-1).argmax(-1),
                'aux_ADS':1-aux[0].argmax(-1), 'aux_CN':aux[1].argmax(-1)}
            for name,module in [('own_only',model.Message_MLP),('semantic_only',model.Global_Message)]:
                handle=module.register_forward_hook(lambda mod,args,out:torch.zeros_like(out))
                try:all_pred[name]=model(data['Feature'])[0].argmax(-1)
                finally:handle.remove()
            row=dict(fold=fold,epoch=cp['best']['epoch'],checkpoint_sha256=sha(folder/'best.pt'),metrics={})
            for name,pred in all_pred.items():
                row['metrics'][name]={split:int((pred[mask]==y[mask]).sum())/int(mask.sum()) for split,mask in [('train',fit),('test',test)]}
                row['metrics'][name]['corrected_vs_main']=int(((pred[test]==y[test])&(all_pred['main'][test]!=y[test])).sum())
                row['metrics'][name]['harmed_vs_main']=int(((pred[test]!=y[test])&(all_pred['main'][test]==y[test])).sum())
            records.append(row)
    means={name:{split:float(np.mean([r['metrics'][name][split] for r in records])) for split in ['train','test']} for name in records[0]['metrics']}
    result=dict(records=records,means=means,script_sha256=sha(__file__),scope='Fixed existing W test-selected best checkpoints, CPU eval with exact argmax replay. No fitting or threshold search. Zeroing a branch is a post-hoc distribution intervention, not a trained branch ablation. Auxiliary outputs already trained on train masks; reading them does not establish superiority of a separately trained head.')
    dump(ROOT/'Result/abide_c_forensic_20260927/W_readout_audit.json',result)
    print(json.dumps(means))

if __name__=='__main__':main()
