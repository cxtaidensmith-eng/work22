"""Cooperative-sheaf variants under the RETAINED ChebGCN protocol.

Original test-ACC checkpoint selection is deliberately retained at the user's
request. These are exploratory comparisons, not independent test estimates.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from Utils import Config_, SET_Random, load_path, load_dataset, CustomCosineAnnealingLR
from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest

ROOT = Path(__file__).resolve().parent
MODES = ('shared', 'content', 'static', 'dynamic', 'self_only')
FIRST_ROUND_MODES = ('shared', 'static', 'dynamic')


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def training_seed_for(args, cfg):
    seed = getattr(args, 'training_seed', None)
    return cfg.seed * 100 if seed is None else seed


def setup(cuda=0, config='Config/main_tadpole_3class.ini', k=10):
    torch.set_num_threads(1)
    cfg = Config_(str(ROOT), str(ROOT/config), cuda)
    if cfg.n_seeds != 1 or cfg.use_ema or cfg.mixup_alpha != 0 or cfg.gate_sparsity_lambda != 0:
        raise ValueError('Runner is for the retained single-seed/no-EMA/no-mixup/no-gate-loss protocol')
    if cfg.train_size != 0:
        raise ValueError('Requires the retained ten-fold protocol')
    SET_Random(cfg.seed)
    fp, dp, _, names = load_path(str(ROOT), cfg.DATA_SET, cfg.Task)
    dd, data = load_dataset(fp, dp, cfg.Device, names, cfg.Shuffle, cfg.seed, train_size=cfg.train_size)
    if cfg.DATA_SET == 'TADPOLE' and cfg.Task == 'AD_CN_SMCI' and cfg.seed == 0:
        with np.load(ROOT/'reference_fixed10/data_and_splits.npz') as historical:
            assert np.array_equal(historical['source_row'], dd['Index'])
            assert np.array_equal(historical['labels'], data['Label'].cpu().numpy())
            for f, (_, mask) in enumerate(data['Mask']):
                assert np.array_equal(historical['test_masks'][f], mask.cpu().numpy())
    graphs = modal_candidates(data['Feature'], dd['Modal_Index'], k)
    return cfg, dd, data, graphs, [fp, dp]


def scores(logits, labels):
    z, y = logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
    p = z.argmax(-1)
    classes = z.shape[-1]
    # Match main.py: raw-logit one-vs-rest macro AUC and weighted F1.
    return dict(acc=float((p == y).mean()),
                auc=float(roc_auc_score(np.eye(classes)[y], z)),
                f1=float(f1_score(y, p, average='weighted', zero_division=0)),
                macro_f1=float(f1_score(y, p, average='macro', zero_division=0)),
                per_class_recall=recall_score(y, p, labels=list(range(classes)), average=None,
                                             zero_division=0).tolist(),
                confusion_matrix=confusion_matrix(y, p, labels=list(range(classes))).tolist())


def build_protocol(args, cfg, dd, data, graphs, paths):
    config_keys = ('DATA_SET', 'Task', 'seed', 'train_size', 'Shuffle', 'Hidden_size', 'Drop_rate',
                   'num_layers', 'num_heads', 'input_noise_std', 'drop_path', 'global_word_emb',
                   'semantic_branch', 'modal_structure', 'readout_mode', 'lr', 'weight_decay',
                   'Loss_rate', 'aux_loss_weight', 'grad_clip', 'Use_scheduler', 'T_max', 'Lr_Min',
                   'n_seeds', 'use_ema', 'mixup_alpha', 'gate_sparsity_lambda')
    sources = ['relation_experiment.py', 'Model/relation_sheaf.py', 'Model/network.py',
               'Model/models.py', 'Utils/utils.py', 'Utils/data_load.py', 'Utils/graph_load.py',
               'Loss/loss_fn.py', args.config]
    if (ROOT/'relation_followup.py').exists():
        sources.append('relation_followup.py')
    for source in ('Utils/graph_diagnostics.py', 'relation_audit.py',
                   'Utils/relation_optimizer.py', 'relation_decay.py'):
        if (ROOT/source).exists():
            sources.append(source)
    return dict(experiment='relation-content cooperative sheaf controlled experiment',
                modes=args.modes, folds=args.folds, epochs=args.epochs or cfg.epochs,
                split_seed=cfg.seed, training_seed=training_seed_for(args, cfg),
                diagnostics=bool(getattr(args, 'diagnostics', False)),
                diagnostic_schedule='Epochs 1-20, every tenth epoch, and final epoch; passive hooks' if getattr(args, 'diagnostics', False) else None,
                smoke=bool(args.epochs and args.epochs != cfg.epochs),
                original_config={key: getattr(cfg, key) for key in config_keys},
                graph_weight_decay_override=getattr(args, 'graph_weight_decay', None),
                graph_weight_decay_scope='propagation.* except output_norm; all other parameters keep original decay',
                checkpoint_selection='First maximum TEST ACC, identical to original n_seeds=1 selection',
                training='Full original train fold; Adam; original scheduler; original cohort class weights',
                labels='Only train-mask labels in gradient losses; test labels used for historical selection',
                transductive=True, preprocessing='Existing processed CSV unchanged; no refitting of preprocessing',
                unsupervised_regularizer='Original all-cohort class-embedding orthogonality loss retained',
                candidate_k=args.k, support='Per-modal cosine top-k, stable CPU sort, union with reverse edges',
                candidate_hash=hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest(),
                graph_statistics=[dict(modality=name, directed_edges=int(g['edge'].shape[1]),
                    degree_min=int(g['degree'].min()), degree_mean=float(g['degree'].float().mean()),
                    degree_max=int(g['degree'].max())) for name, g in zip(dd['Modal_Name'], graphs)],
                operator='Paper Eqs 7-8: Delta_in^T Delta_out, weighted symmetric support, D+I normalization',
                graph_layers=1, graph_passes=2, stalk_dimension=2,
                mode_overrides={'self_only': 'No graph propagation; retain output LayerNorm, encoder and readout; all graph-only parameters removed'},
                maps='2D rotations times bounded positive scales; shared/content(n)/static(m,n)/dynamic(m,n,node-content)',
                common_weights='Seeded identically including MLP head and all common propagation components',
                parameter_matching='Content-only and static have equal active parameter counts; other variants differ',
                metrics='ACC; original weighted F1 and raw-logit OVR macro AUC; supplementary macro F1 and recalls',
                class_names=dd['Class_Names'], samples=dd['Sample_Num'], modalities=dd['Modal_Name'],
                device=str(cfg.Device), torch=torch.__version__, cuda=torch.version.cuda,
                code_sha256={s: sha(ROOT/s) for s in sources},
                data_sha256={Path(p).name: sha(p) for p in paths},
                caveat='Historical test-selected estimates; hypothesis generation only, no independent confirmation')


def update_summary(out, args):
    summaries = {}
    records = {}
    for mode in args.modes:
        rows = []
        for fold in args.folds:
            file = out/mode/f'fold_{fold:02d}'/'complete.json'
            if file.exists():
                rows.append(json.loads(file.read_text(encoding='utf-8')))
        records[mode] = rows
        summaries[mode] = dict(completed_folds=len(rows), planned_folds=len(args.folds))
        if rows:
            summaries[mode].update(parameters=rows[0]['parameters'],
                mean_acc=float(np.mean([r['best']['acc'] for r in rows])),
                std_acc=float(np.std([r['best']['acc'] for r in rows])),
                mean_final_epoch_acc=float(np.mean([r['last']['acc'] for r in rows])),
                mean_auc=float(np.mean([r['best']['auc'] for r in rows])),
                mean_f1=float(np.mean([r['best']['f1'] for r in rows])),
                mean_macro_f1=float(np.mean([r['best']['macro_f1'] for r in rows])),
                folds=[dict(fold=r['fold'], epoch=r['best']['epoch'], acc=r['best']['acc']) for r in rows])
    pairs = []
    for left, right in [('shared', 'static'), ('content', 'static'), ('shared', 'content'), ('static', 'dynamic')]:
        if left not in records or right not in records:
            continue
        common = sorted({r['fold'] for r in records[left]} & {r['fold'] for r in records[right]})
        corrected = harmed = 0
        deltas = []
        for fold in common:
            with np.load(out/left/f'fold_{fold:02d}'/'predictions.npz') as a, \
                 np.load(out/right/f'fold_{fold:02d}'/'predictions.npz') as b:
                assert np.array_equal(a['labels'], b['labels']) and np.array_equal(a['row_index'], b['row_index'])
                ca, cb = a['logits'].argmax(-1) == a['labels'], b['logits'].argmax(-1) == b['labels']
                corrected += int((~ca & cb).sum())
                harmed += int((ca & ~cb).sum())
                deltas.append(float(cb.mean() - ca.mean()))
        if common:
            pairs.append(dict(comparison=f'{right} vs {left}', matched_folds=common,
                              mean_acc_delta=float(np.mean(deltas)), corrected=corrected, harmed=harmed))
    result = dict(updated_at=timestamp(), variants=summaries, matched_comparisons=pairs,
                  caveat='First maximum test ACC; partial folds are not the final ten-fold result')
    dump(out/'summary.json', result)
    names = {'shared': 'G1：统一信息变换', 'content': 'C：只按内容模态变换（与 G2 等参数量）',
             'static': 'G2：按关系模态与内容模态变换',
             'dynamic': 'G3：进一步结合患者状态动态变换',
             'self_only': 'S：从训练开始不使用患者间传播'}
    protocol = json.loads((out/'PROTOCOL.json').read_text(encoding='utf-8'))
    lines = ['# 关系与内容匹配实验：当前进度', '',
             f"数据：{protocol['original_config']['DATA_SET']} / {protocol['original_config']['Task']}；"
             f"每折 {protocol['epochs']} 轮，训练 seed={protocol.get('training_seed', 0)}，原十折划分固定。",
             '各折取首次达到最高测试 ACC 的检查点；这是原协议下的探索性结果。', '',
             '以下均值只覆盖已经完成的折，三个版本折数不同时请以配对比较为准。', '']
    for mode, item in summaries.items():
        line = f"- {names[mode]}：完成 {item['completed_folds']}/{item['planned_folds']} 折"
        if item['completed_folds']:
            line += f"，平均 ACC {100*item['mean_acc']:.2f}%，参数量 {item['parameters']:,}。"
        lines.append(line)
    if pairs:
        lines.extend(['', '相同已完成折上的配对比较：', ''])
        for pair in pairs:
            lines.append(f"- {pair['comparison']}：{len(pair['matched_folds'])} 折，"
                         f"ACC 差值 {100*pair['mean_acc_delta']:+.2f} 个百分点，"
                         f"纠正 {pair['corrected']} 人，新增错误 {pair['harmed']} 人。")
    lines.extend(['', '机制比较须等待计划中的版本和折全部完成；不能用单折或最佳中间数值判断机制有效。',
                  '', '逐折指标、选中轮次及固定末轮结果见 summary.json；运行位置见 status.json。'])
    (out/'进度与结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def run_fold(out, mode, fold, args, cfg, dd, data, graphs):
    folder = out/mode/f'fold_{fold:02d}'
    if (folder/'complete.json').exists():
        print(f'Skip completed {mode} fold {fold}', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    seed = training_seed_for(args, cfg)
    SET_Random(seed)
    model = RelationSheafModel(dd, cfg, graphs, mode,
                              training_seed=getattr(args, 'training_seed', None)).to(cfg.Device)
    parameters = sum(p.numel() for p in model.parameters())
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, getattr(args, 'graph_weight_decay', None))
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min) if cfg.Use_scheduler else None
    x, y = data['Feature'], data['Label']
    fit, test = data['Mask'][fold-1]
    epochs = args.epochs or cfg.epochs
    best_correct, best, last = -1, None, None
    started = time.perf_counter()
    if cfg.Device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(cfg.Device)
    fields = ['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr', 'seconds']
    observer = None
    if getattr(args, 'diagnostics', False):
        from Utils.graph_diagnostics import GraphDiagnostics
        observer = GraphDiagnostics(model, folder/'diagnostics.csv', cfg.weight_decay, optimizer=optimizer)
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream, ExitStack() as cleanup:
        if observer is not None:
            cleanup.callback(observer.close)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs+1):
            if observer is not None:
                observer.begin(epoch, epochs, optimizer.param_groups[0]['lr'])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = criterion(logits, y, fit, embeddings, auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} fold {fold} epoch {epoch}: nonfinite training loss')
            loss.backward()
            if observer is not None:
                observer.after_backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            if observer is not None:
                observer.before_step(grad_norm)
            optimizer.step()
            if observer is not None:
                observer.after_step()
            model.eval()
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                       test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'],
                       seconds=time.perf_counter()-started)
            writer.writerow(row)
            if observer is not None:
                observer.finish(row['train_loss'], row['test_loss'], row['test_acc'])
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **scores(logits[test], y[test]))
                checkpoint = folder/'best.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), best=best, mode=mode, fold=fold,
                                protocol='original test-selected', seed=seed), checkpoint)
                checkpoint.replace(folder/'best.pt')
                np.savez_compressed(folder/'predictions.npz', logits=logits[test].cpu().numpy(),
                    labels=y[test].cpu().numpy(), row_index=np.array(dd['Index'])[test.cpu().numpy()])
                dump(folder/'best.json', best)
            if scheduler is not None:
                scheduler.step()
            if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), mode=mode, fold=fold, seed=seed,
                              epochs=epochs, parameters=parameters, best_acc=best['acc'], **row)
                dump(out/'status.json', status)
                print(json.dumps(status), flush=True)
            if epoch == epochs:
                last = dict(epoch=epoch, **scores(logits[test], y[test]))
                np.savez_compressed(folder/'last_predictions.npz', logits=logits[test].cpu().numpy(),
                    labels=y[test].cpu().numpy(), row_index=np.array(dd['Index'])[test.cpu().numpy()])
    record = dict(mode=mode, fold=fold, seed=seed, parameters=parameters, best=best, last=last,
                  completed_at=timestamp(), seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated(cfg.Device)/2**20 if cfg.Device.type == 'cuda' else None)
    if observer is not None:
        dump(folder/'diagnostics_summary.json', observer.summary())
    dump(folder/'complete.json', record)
    print(json.dumps(dict(event='fold_complete', **record)), flush=True)
    del model, optimizer, criterion, logits, embeddings, auxiliary, loss
    if cfg.Device.type == 'cuda':
        torch.cuda.empty_cache()


def train(args, progress_callback=None):
    out = (ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume to restart incomplete folds')
    cfg, dd, data, graphs, paths = setup(args.cuda, args.config, args.k)
    protocol = build_protocol(args, cfg, dd, data, graphs, paths)
    reference = getattr(args, 'reference_protocol', None)
    if reference is not None:
        legacy = json.loads(Path(reference).read_text(encoding='utf-8'))
        for key in ('original_config', 'epochs', 'candidate_hash', 'data_sha256', 'samples', 'class_names'):
            if legacy[key] != protocol[key]:
                raise RuntimeError(f'Original protocol changed: {key}')
        for source, digest in legacy['code_sha256'].items():
            if source not in ('Model/relation_sheaf.py', 'relation_experiment.py'):
                if protocol['code_sha256'][source] != digest:
                    raise RuntimeError(f'Original component changed: {source}')
    if (out/'PROTOCOL.json').exists():
        old = json.loads((out/'PROTOCOL.json').read_text(encoding='utf-8'))
        if old != protocol:
            raise RuntimeError('Protocol/code/data changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'PROTOCOL.json', protocol)
    np.savez_compressed(out/'splits_and_graphs.npz', source_row=np.array(dd['Index']),
        labels=data['Label'].cpu().numpy(),
        test_masks=np.stack([mask.cpu().numpy() for _, mask in data['Mask']]),
        **{f'edge_{m}': g['edge'].numpy() for m, g in enumerate(graphs)},
        **{f'neighbors_{m}': g['neighbors'].numpy() for m, g in enumerate(graphs)})
    update_summary(out, args)
    try:
        for fold in args.folds:
            for mode in args.modes:
                run_fold(out, mode, fold, args, cfg, dd, data, graphs)
                update_summary(out, args)
                if progress_callback is not None:
                    progress_callback()
        dump(out/'status.json', dict(state='complete', updated_at=timestamp()))
        dump(out/'COMPLETE.json', update_summary(out, args))
    except BaseException as error:
        dump(out/'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='Config/main_tadpole_3class.ini')
    parser.add_argument('--output', default='Result/relation_sheaf_tadpole_original')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(FIRST_ROUND_MODES))
    parser.add_argument('--folds', type=int, choices=range(1, 11), nargs='+', default=list(range(1, 11)))
    parser.add_argument('--epochs', type=int, default=None, help='Smoke test override only; default uses original config')
    parser.add_argument('--training-seed', type=int, default=None, help='Training RNG only; never changes data order or splits')
    parser.add_argument('--diagnostics', action='store_true', help='Passively trace graph updates, gradients, weight decay and weights')
    parser.add_argument('--graph-weight-decay', type=float, default=None,
                        help='Explicit ablation: override graph-only decay; retain patient-path decay')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('epochs must be positive')
    if args.training_seed is not None and args.training_seed < 0:
        parser.error('training-seed must be nonnegative')
    if len(set(args.folds)) != len(args.folds) or len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate folds or modes are not allowed')
    train(args)
