"""Verify or reproduce the retained fixed-10 BB reference model only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from Utils import Config_, SET_Random, load_path, load_dataset, CustomCosineAnnealingLR
from Loss import MSCQGNLoss
from Model.fixed_reference import FixedNeighborReference, fixed_candidates, reference_loss

ROOT = Path(__file__).resolve().parent
RETAINED = ROOT / 'reference_fixed10'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(logits, labels):
    z, y = logits.detach().cpu().numpy(), labels.cpu().numpy()
    return dict(acc=float(accuracy_score(y, z.argmax(-1))),
                auc=float(roc_auc_score(np.eye(3)[y], z, average='macro')),
                macro_f1=float(f1_score(y, z.argmax(-1), average='macro')))


def setup(cuda):
    torch.set_num_threads(1)
    cfg = Config_(str(ROOT), str(ROOT/'Config/main_tadpole_3class.ini'), cuda)
    SET_Random(0)
    fp, dp, _, names = load_path(str(ROOT), cfg.DATA_SET, cfg.Task)
    dd, data = load_dataset(fp, dp, cfg.Device, names, cfg.Shuffle, cfg.seed, train_size=cfg.train_size)
    candidates = fixed_candidates(data['Feature'])
    with np.load(RETAINED/'data_and_splits.npz') as old:
        assert np.array_equal(old['source_row'], dd['Index'])
        assert np.array_equal(old['labels'], data['Label'].cpu().numpy())
        assert np.array_equal(old['candidates'], candidates.cpu().numpy())
        assert all(np.array_equal(old['test_masks'][f], mask.cpu().numpy()) for f, (_, mask) in enumerate(data['Mask']))
    return cfg, dd, data, candidates


def verify(args):
    cfg, dd, data, candidates = setup(args.cuda)
    x, y = data['Feature'], data['Label']
    records = []
    for fold, (_, mask) in enumerate(data['Mask'], 1):
        folder = RETAINED/'checkpoints'/f'fold_{fold:02d}'
        SET_Random(0)
        model = FixedNeighborReference(dd, cfg, candidates).to(cfg.Device).eval()
        model.load_historical_state(torch.load(folder/'best.pt', map_location=cfg.Device, weights_only=True)['state_dict'])
        with torch.inference_mode():
            r = model(x)
        with np.load(folder/'best_predictions.npz') as cache:
            error = float(np.max(np.abs(r['logits'].cpu().numpy()-cache['logits'])))
            assert error < 2e-4
            assert np.array_equal(r['logits'].argmax(-1).cpu().numpy(), cache['logits'].argmax(-1))
            assert np.array_equal(r['self_logits'].argmax(-1).cpu().numpy(), cache['self_logits'].argmax(-1))
        own, final = r['self_logits'][mask].argmax(-1), r['logits'][mask].argmax(-1)
        records.append(dict(fold=fold, max_logit_error=error, self=metrics(r['self_logits'][mask], y[mask]),
            final=metrics(r['logits'][mask], y[mask]), corrected=int(((own!=y[mask])&(final==y[mask])).sum()),
            harmed=int(((own==y[mask])&(final!=y[mask])).sum())))
    summary = dict(folds=10, patients=len(x), self_acc=float(np.mean([r['self']['acc'] for r in records])),
        final_acc=float(np.mean([r['final']['acc'] for r in records])), corrected=sum(r['corrected'] for r in records),
        harmed=sum(r['harmed'] for r in records), max_logit_error=max(r['max_logit_error'] for r in records), records=records,
        caveat='Historical first maximum test ACC; frozen replay, not independent validation')
    dump(RETAINED/'verification.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}, ensure_ascii=False, indent=2))


def train(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    cfg, dd, data, candidates = setup(args.cuda)
    x, y = data['Feature'], data['Label']
    dump(out/'PROTOCOL.json', dict(model='fixed10 BB only', epochs=args.epochs, folds=args.folds, seed=0,
        checkpoint='first maximum test ACC (historical reproduction); fixed last epoch also saved',
        note='No test-driven hyperparameter search; historical selection must be disclosed'))
    for fold, (fit, test) in enumerate(data['Mask'][:args.folds], 1):
        folder = out/f'fold_{fold:02d}'; folder.mkdir()
        SET_Random(0)
        model = FixedNeighborReference(dd, cfg, candidates).to(cfg.Device)
        criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
        best_acc, rows = -1., []
        for epoch in range(1,args.epochs+1):
            model.train(); optimizer.zero_grad()
            r=model(x); loss=reference_loss(criterion,r,y,fit)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step(); model.eval()
            with torch.no_grad(): r=model(x)
            row=dict(epoch=epoch,train_loss=float(loss.detach()),**metrics(r['logits'][test],y[test]))
            rows.append(row)
            if row['acc']>best_acc:
                best_acc=row['acc']; torch.save(dict(state_dict=model.state_dict(),record=row),folder/'best.pt')
                dump(folder/'best.json',row)
                np.savez_compressed(folder/'best_predictions.npz',**{k:r[k].cpu().numpy() for k in ('logits','self_logits','modal_embedding','channel_weights','message_values','channel_contributions')})
            scheduler.step()
            if epoch==1 or epoch%50==0: print(json.dumps(dict(fold=fold,**row)),flush=True)
        torch.save(dict(state_dict=model.state_dict(),record=row),folder/'last.pt')
        with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        dump(folder/'complete.json',dict(best_acc=best_acc,last=row))
    dump(out/'COMPLETE.json',dict(folds=args.folds,epochs=args.epochs))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','train'])
    p.add_argument('--cuda',type=int,default=0)
    p.add_argument('--output',default='Result/fixed10_new')
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--folds',type=int,choices=range(1,11),default=10)
    args=p.parse_args()
    if args.epochs<1: p.error('--epochs must be positive')
    verify(args) if args.action=='verify' else train(args)
