"""Read historical Git artifacts; never restore files or train a model."""
import ast
import hashlib
import io
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Result/abide_c_forensic_20260927'
REF=ROOT/'Result/c_abide_repair_v2_20260927/full/W'

def show(path,ref='HEAD'):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT.parent)

def main():
    csv=pd.read_csv(ROOT/'DATASET/ABIDE/ABIDE_processed_data_modal.csv')
    with np.load(REF/'splits_and_graphs.npz') as p:
        rows=p['source_row']; labels=p['labels']; masks=p['test_masks']
    raw=csv.iloc[:,:-1].to_numpy(dtype=np.float32)[rows]
    records=[]
    for fold,selection in [(0,'test_acc'),(8,'test_acc'),(0,'last')]:
        prefix=f'Visualization/tsne_paper/ABIDE_ADS_CN_ABIDE_cheb_h64_d045_k3_fold{fold}_ep1000_{selection}'
        blob=show(prefix+'/embeddings_and_meta.npz')
        with np.load(io.BytesIO(blob)) as cache:
            lookup={row.tobytes():i for i,row in enumerate(cache['Raw_Input'])}
            assert len(lookup)==len(raw)
            perm=np.array([lookup[row.tobytes()] for row in raw])
            np.testing.assert_array_equal(cache['Raw_Input'][perm],raw)
            np.testing.assert_array_equal(cache['label'][perm],labels)
            test=cache['test_mask'][perm].astype(bool); train=cache['train_mask'][perm].astype(bool)
            np.testing.assert_array_equal(test,masks[fold]); np.testing.assert_array_equal(train,~masks[fold])
            pred=cache['Logits'][perm].argmax(1)
            np.testing.assert_array_equal(pred,cache['pred'][perm])
            records.append(dict(fold=fold+1,selection=selection,git_path=prefix,sha256=hashlib.sha256(blob).hexdigest(),
                cache_keys=list(cache.keys()),train_correct=int((pred[train]==labels[train]).sum()),train_n=int(train.sum()),
                test_correct=int((pred[test]==labels[test]).sum()),test_n=int(test.sum()),
                train_acc=float((pred[train]==labels[train]).mean()),test_acc=float((pred[test]==labels[test]).mean()),
                exact_data_label_mask_match=True,summary=show(prefix+'/summary.txt').decode()))
    script=show('scripts/tsne_visualization.py').decode('utf-8-sig')
    tree=ast.parse(script)
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run_training')
    calls=[ast.unparse(node.func) for node in ast.walk(fn) if isinstance(node,ast.Call)]
    cfg=show('Config/ABIDE_cheb_h64_d045_k3.ini','ed3cea6').decode('utf-8-sig')
    (OUT/'historical_formal_config.ini').write_text(cfg,encoding='utf-8')
    (OUT/'historical_tsne_run_training.py.txt').write_text(ast.get_source_segment(script,fn),encoding='utf-8')
    result=dict(records=records,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        available_visualization_generator=dict(sha256=hashlib.sha256(script.encode()).hexdigest(),calls=calls,
            scheduler_called=any('schedul' in call.lower() for call in calls),trains_fresh_model='build_model' in calls),
        interpretation='Caches contain predictions from visualization reruns, not proven artifacts of formal tenfold run20260508. Available generator has no LR scheduler; frozen formal config enables scheduler. Cache has no source hash or LR trace, so exact generator version/settings at creation cannot be established. Cache confirms input/label/mask identity and corresponding prediction accuracy only. Two-fold representation comparison is descriptive, not controlled attribution of architecture or the formal90.59% gap.')
    (OUT/'historical_provenance_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(records=[{k:r[k] for k in ['fold','selection','train_acc','test_acc']} for r in records],generator=result['available_visualization_generator']),ensure_ascii=False))

if __name__=='__main__': main()
