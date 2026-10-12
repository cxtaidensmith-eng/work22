"""Four fixed-protocol controls of content-modal reference selection.

The outer historical TEST-ACC selection is deliberately retained. Teachers use
only inner-fit labels, a fixed last epoch, and held-out TRAIN rows for targets.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from relation_experiment import ROOT, setup, dump, sha, timestamp, scores, build_protocol
from Utils import SET_Random, CustomCosineAnnealingLR
from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel
from Model.reference_selection import ReferenceSelectionModel
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from Utils.reference_selection import (rewire_candidates, inner_splits, coalition_masks,
    mean_marginal_gain, training_class_weights, gain_supervision_loss)

MODES = ('c_real', 'c_random', 'gate', 'gain_gate')
NAMES = dict(c_real='原 C', c_random='C：随机邻居', gate='普通参照门控', gain_gate='收益监督参照门控')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def folder_for(out, seed, mode, fold):
    return out/f'seed_{seed}'/mode/f'fold_{fold:02d}'


def status(out, **fields):
    value = dict(updated_at=timestamp(), **fields)
    dump(out/'status.json', value)
    print(json.dumps(value), flush=True)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optim_and_loss(model, cfg, dd, folder):
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, graph_weight_decay=0.)
    dump(folder/'optimizer_groups.json', optimizer_manifest(model, optimizer))
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min) if cfg.Use_scheduler else None
    return criterion, optimizer, scheduler


def step(model, optimizer, loss, cfg):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()


def prediction_file(path, logits, y, test, dd, model):
    extra = {}
    if isinstance(model, ReferenceSelectionModel):
        extra['gates'] = model.propagation.last_gate[test].cpu().numpy()
    np.savez_compressed(path, logits=logits[test].cpu().numpy(), labels=y[test].cpu().numpy(),
        row_index=np.asarray(dd['Index'])[test.cpu().numpy()], **extra)


def metric(logits, labels):
    return dict(**scores(logits, labels), nll=float(F.cross_entropy(logits, labels)))


def outer_fold(out, args, cfg, dd, data, graphs, seed, fold, mode, gains=None):
    folder = folder_for(out, seed, mode, fold)
    if (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    SET_Random(seed)
    gated = mode in ('gate', 'gain_gate')
    model = (ReferenceSelectionModel(dd, cfg, graphs, seed) if gated else
             RelationSheafModel(dd, cfg, graphs, 'content', training_seed=seed)).to(cfg.Device)
    initial_hash = model_hash(model)
    criterion, optimizer, scheduler = optim_and_loss(model, cfg, dd, folder)
    fit, test = data['Mask'][fold-1]
    x, y = data['Feature'], data['Label']
    if mode == 'gain_gate':
        if gains is None or not torch.isfinite(gains[fit]).all() or not torch.isnan(gains[test]).all():
            raise ValueError('Gain targets must exist only on outer training rows')
    best_correct, best = -1, None
    started = time.perf_counter()
    history = []
    if cfg.Device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(cfg.Device)
    fields = ['epoch', 'train_loss', 'main_loss', 'gain_loss', 'test_loss', 'test_acc',
              'gate_mean', 'gate_zero_fraction', 'gate_one_fraction', 'lr', 'seconds']
    with (folder/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            main_loss = criterion(logits, y, fit, embeddings, auxiliary)
            extra_loss = (gain_supervision_loss(model.propagation.last_gate_logits, gains, fit,
                          args.gain_temperature) if mode == 'gain_gate' else main_loss.new_zeros(()))
            loss = main_loss + args.gain_weight * extra_loss
            step(model, optimizer, loss, cfg)
            model.eval()
            with torch.no_grad():
                logits, embeddings, auxiliary = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite evaluation logits')
                test_loss = criterion(logits, y, test, embeddings, auxiliary)
            correct = int((logits[test].argmax(-1) == y[test]).sum())
            gates = model.propagation.last_gate if gated else None
            row = dict(epoch=epoch, train_loss=float(loss.detach()), main_loss=float(main_loss.detach()),
                gain_loss=float(extra_loss.detach()), test_loss=float(test_loss), test_acc=correct/int(test.sum()),
                gate_mean=float(gates.mean()) if gated else 1.,
                gate_zero_fraction=float((gates == 0).float().mean()) if gated else 0.,
                gate_one_fraction=float((gates == 1).float().mean()) if gated else 1.,
                lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            writer.writerow(row)
            history.append(row['test_acc'])
            if correct > best_correct:
                best_correct = correct
                best = dict(epoch=epoch, **metric(logits[test], y[test]))
                if gated:
                    best['mean_gate_by_modality'] = gates.mean(0).cpu().tolist()
                    best['test_gate_by_modality'] = gates[test].mean(0).cpu().tolist()
                tmp = folder/'best.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), best=best, mode=mode, fold=fold, seed=seed), tmp)
                tmp.replace(folder/'best.pt')
                prediction_file(folder/'predictions.npz', logits, y, test, dd, model)
                dump(folder/'best.json', best)
            if scheduler:
                scheduler.step()
            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                stream.flush()
                status(out, state='running', stage='outer', mode=mode, seed=seed, fold=fold,
                       epochs=args.epochs, best_acc=best['acc'], **row)
        last = dict(epoch=args.epochs, **metric(logits[test], y[test]))
        if gated:
            last['mean_gate_by_modality'] = gates.mean(0).cpu().tolist()
        prediction_file(folder/'last_predictions.npz', logits, y, test, dd, model)
    record = dict(mode=mode, seed=seed, fold=fold, best=best, last=last,
        parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial_hash,
        last_100_mean_acc=float(np.mean(history[-100:])), completed_at=timestamp(),
        seconds=time.perf_counter()-started,
        peak_allocated_mb=torch.cuda.max_memory_allocated(cfg.Device)/2**20 if cfg.Device.type == 'cuda' else None)
    dump(folder/'complete.json', record)


def teacher_targets(out, args, cfg, dd, data, graphs, seed, fold):
    folder = folder_for(out, seed, 'crossfit', fold)
    folder.mkdir(parents=True, exist_ok=True)
    fit, test = data['Mask'][fold-1]
    x, y = data['Feature'], data['Label']
    m = len(dd['Modal_Name'])
    if (folder/'complete.json').exists():
        with np.load(folder/'targets.npz') as saved:
            return torch.as_tensor(saved['gains'], device=cfg.Device)
    split_seed = 31009 + seed*100 + fold
    partitions = inner_splits(y, fit, test, split_seed, n_splits=2)
    masks = coalition_masks(m)
    gains = np.full((len(y), m), np.nan, dtype=np.float32)
    assignment = np.full(len(y), -1, dtype=np.int32)
    for inner, (train_np, held_np) in enumerate(partitions):
        sub = folder/f'inner_{inner}'
        sub.mkdir(parents=True, exist_ok=True)
        if not (sub/'complete.json').exists():
            teacher_seed = 52001 + seed*100 + fold*2 + inner
            SET_Random(teacher_seed)
            model = ReferenceSelectionModel(dd, cfg, graphs, teacher_seed, learned=False).to(cfg.Device)
            train = torch.as_tensor(train_np, device=cfg.Device)
            held = torch.as_tensor(held_np, device=cfg.Device)
            teacher_dd = dict(dd, Label_Weight=training_class_weights(y, train, dd['Class_Num']))
            criterion, optimizer, scheduler = optim_and_loss(model, cfg, teacher_dd, sub)
            # Held-out and outer-test labels are replaced before calling the
            # teacher training criterion, including its auxiliary one-hot step.
            teacher_y = torch.where(train, y, torch.zeros_like(y))
            generator = torch.Generator(device=cfg.Device).manual_seed(teacher_seed + 9100)
            started = time.perf_counter()
            with (sub/'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['epoch', 'train_loss', 'lr', 'seconds'])
                for epoch in range(1, args.teacher_epochs+1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    # Per-patient mixture: 25% all off, 25% all on, 50% random
                    # subsets. Gates are downstream of the one graph operator.
                    group = torch.rand(len(y), 1, generator=generator, device=cfg.Device)
                    mask = (torch.rand(len(y), m, generator=generator, device=cfg.Device) < .5).float()
                    mask = torch.where(group < .25, torch.zeros_like(mask), mask)
                    mask = torch.where(group >= .75, torch.ones_like(mask), mask)
                    logits, embeddings, auxiliary = model(x, mask)
                    loss = criterion(logits, teacher_y, train, embeddings, auxiliary)
                    step(model, optimizer, loss, cfg)
                    writer.writerow([epoch, float(loss.detach()), optimizer.param_groups[0]['lr'],
                                     time.perf_counter()-started])
                    if scheduler:
                        scheduler.step()
                    if epoch == 1 or epoch % 25 == 0 or epoch == args.teacher_epochs:
                        stream.flush()
                        status(out, state='running', stage='crossfit_teacher', seed=seed, fold=fold,
                            inner=inner, epoch=epoch, epochs=args.teacher_epochs, train_loss=float(loss.detach()),
                            seconds=time.perf_counter()-started)
            # Fixed LAST epoch, no validation/test checkpoint selection. Only
            # the inner held-out TRAIN labels are used for these loss targets.
            model.eval()
            with torch.no_grad():
                losses = np.stack([F.cross_entropy(model(x, mask[None])[0][held], y[held], reduction='none')
                    .cpu().numpy() for mask in masks])
            if not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite teacher coalition losses')
            np.savez_compressed(sub/'coalitions.npz', losses=losses, masks=masks,
                train_mask=train_np, held_mask=held_np, row_index=np.asarray(dd['Index'])[held_np])
            dump(sub/'complete.json', dict(seed=teacher_seed, inner=inner, epochs=args.teacher_epochs,
                train_count=int(train_np.sum()), held_count=int(held_np.sum()),
                inner_train_class_weights=teacher_dd['Label_Weight'].cpu().tolist(),
                selection='fixed last epoch; no held-out/test evaluation during training',
                state_sha256=model_hash(model), artifact_sha256=sha(sub/'coalitions.npz'),
                seconds=time.perf_counter()-started, completed_at=timestamp()))
            del model, optimizer, criterion, logits, embeddings, auxiliary, loss
        with np.load(sub/'coalitions.npz') as saved:
            np.testing.assert_array_equal(saved['train_mask'], train_np)
            np.testing.assert_array_equal(saved['held_mask'], held_np)
            gains[held_np] = mean_marginal_gain(saved['losses'], m)
        assignment[held_np] = inner
    fit_np, test_np = fit.cpu().numpy(), test.cpu().numpy()
    assert np.isfinite(gains[fit_np]).all() and np.isnan(gains[test_np]).all()
    np.savez_compressed(folder/'targets.npz', gains=gains, fit_mask=fit_np, test_mask=test_np,
                        assignment=assignment, source_row=np.asarray(dd['Index']))
    train_gains = gains[fit_np]
    record = dict(seed=seed, fold=fold, split_seed=split_seed, inner_folds=2,
        teacher_epochs=args.teacher_epochs, modalities=dd['Modal_Name'],
        mean_gain=train_gains.mean(0).tolist(), std_gain=train_gains.std(0).tolist(),
        positive_fraction=(train_gains > 0).mean(0).tolist(),
        absolute_gain_above_001=(np.abs(train_gains) > .01).mean(0).tolist(),
        train_rows=int(fit.sum()), test_targets='NaN; excluded from gain training',
        target_sha256=sha(folder/'targets.npz'), completed_at=timestamp(),
        definition='Unweighted held-out NLL(omit)-NLL(include), uniform average over all other-modal subsets')
    dump(folder/'complete.json', record)
    return torch.as_tensor(gains, device=cfg.Device)


def import_baselines(out, args, cfg, dd, data, graphs, protocol):
    """Verify protocol + replay all selected checkpoints; do not duplicate them."""
    if not args.baseline_source:
        return
    for seed in args.seeds:
        source = (ROOT/args.baseline_source/f'seed_{seed}').resolve()
        legacy = read(source/'PROTOCOL.json')
        for key in ('original_config', 'epochs', 'candidate_hash', 'data_sha256', 'samples', 'class_names'):
            if legacy[key] != protocol[key]:
                raise ValueError(f'Baseline protocol mismatch: {key}')
        if legacy['training_seed'] != seed or legacy['graph_weight_decay_override'] != 0:
            raise ValueError('Baseline seed or graph decay mismatch')
        for file, digest in legacy['code_sha256'].items():
            if sha(ROOT/file) != digest:
                raise ValueError(f'Baseline source changed: {file}')
        for fold in args.folds:
            target = folder_for(out, seed, 'c_real', fold)
            if (target/'complete.json').exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            original = source/'content'/f'fold_{fold:02d}'
            record = read(original/'complete.json')
            if record['seed'] != seed or record['fold'] != fold or record['last']['epoch'] != args.epochs:
                raise ValueError('Incomplete/mismatched baseline fold')
            SET_Random(seed)
            model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=seed).to(cfg.Device)
            model.load_state_dict(torch.load(original/'best.pt', map_location=cfg.Device, weights_only=False)['state_dict'])
            model.eval()
            with torch.no_grad():
                logits = model(data['Feature'])[0]
            test = data['Mask'][fold-1][1]
            with np.load(original/'predictions.npz') as saved:
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_allclose(saved['logits'], logits[test].cpu().numpy(), atol=2e-5, rtol=2e-5)
                if abs((saved['logits'].argmax(-1) == saved['labels']).mean() - record['best']['acc']) > 1e-12:
                    raise ValueError('Baseline metrics do not match predictions')
            history = list(csv.DictReader((original/'epochs.csv').open(encoding='utf-8')))
            acc = np.asarray([float(r['test_acc']) for r in history])
            if len(history) != args.epochs or int(acc.argmax())+1 != record['best']['epoch']:
                raise ValueError('Baseline checkpoint is not first maximum historical TEST ACC')
            for file in ('predictions.npz', 'last_predictions.npz', 'optimizer_groups.json'):
                shutil.copy2(original/file, target/file)
            dump(target/'baseline_reference.json', dict(source=str(original), replay_verified=True,
                best_checkpoint_sha256=sha(original/'best.pt'), protocol_sha256=sha(source/'PROTOCOL.json')))
            record.update(mode='c_real', reused_baseline=True, last_100_mean_acc=float(acc[-100:].mean()))
            dump(target/'complete.json', record)
        status(out, state='running', stage='baseline_verified', seed=seed, folds=args.folds)


def summary(out, args):
    records, variants = {}, {}
    for mode in MODES:
        rows = {}
        for seed in args.seeds:
            for fold in args.folds:
                path = folder_for(out, seed, mode, fold)/'complete.json'
                if path.exists():
                    rows[seed, fold] = read(path)
        records[mode] = rows
        variants[mode] = dict(completed=len(rows), planned=len(args.seeds)*len(args.folds))
        if rows:
            variants[mode].update(mean_acc=float(np.mean([r['best']['acc'] for r in rows.values()])),
                mean_last_acc=float(np.mean([r['last']['acc'] for r in rows.values()])),
                mean_last100_acc=float(np.mean([r['last_100_mean_acc'] for r in rows.values()])),
                per_seed={str(s): float(np.mean([r['best']['acc'] for (ss, f), r in rows.items() if ss == s]))
                    for s in args.seeds if any(ss == s for ss, _ in rows)})
    pairs = []
    for left, right in [('c_real', 'c_random'), ('c_real', 'gate'), ('gate', 'gain_gate')]:
        keys = sorted(records[left].keys() & records[right].keys())
        if not keys:
            continue
        deltas, final_deltas, corrected, harmed = [], [], 0, 0
        for seed, fold in keys:
            a, b = records[left][seed, fold], records[right][seed, fold]
            deltas.append(b['best']['acc'] - a['best']['acc'])
            final_deltas.append(b['last']['acc'] - a['last']['acc'])
            with np.load(folder_for(out, seed, left, fold)/'predictions.npz') as pa, \
                 np.load(folder_for(out, seed, right, fold)/'predictions.npz') as pb:
                np.testing.assert_array_equal(pa['row_index'], pb['row_index'])
                np.testing.assert_array_equal(pa['labels'], pb['labels'])
                ca = pa['logits'].argmax(-1) == pa['labels']
                cb = pb['logits'].argmax(-1) == pb['labels']
                corrected += int((~ca & cb).sum()); harmed += int((ca & ~cb).sum())
            if left == 'gate':
                if a['parameters'] != b['parameters'] or a['initial_state_sha256'] != b['initial_state_sha256']:
                    raise ValueError('Gate control is not identical in architecture and initialization')
        pairs.append(dict(comparison=f'{right} vs {left}', matched=len(keys),
            mean_acc_delta=float(np.mean(deltas)), mean_last_acc_delta=float(np.mean(final_deltas)),
            wins=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas), losses=sum(d < 0 for d in deltas),
            corrected_patient_evaluations=corrected, harmed_patient_evaluations=harmed))
    result = dict(updated_at=timestamp(), variants=variants, paired=pairs,
        caveat='Historical first maximum TEST ACC; incomplete folds are not a final result; seeds repeat patients')
    dump(out/'summary.json', result)
    lines = ['# 参照选择实验进度', '', f'原 TADPOLE 十折划分；本次每折 {args.epochs} 轮，首次最高测试 ACC 选轮。',
        '普通门控与收益门控具有完全相同的结构、初始化；只有训练内部收益监督这一项不同。',
        '先判断是否值得按患者选择参照模态；本轮没有选择单个邻居。', '']
    for mode, item in variants.items():
        line = f"- {NAMES[mode]}：{item['completed']}/{item['planned']} 组完成"
        if item['completed']:
            line += f"，已完成组平均 ACC {100*item['mean_acc']:.2f}%。"
        lines.append(line)
    lines += ['', '只比较双方均已完成的相同折与随机种子；不要直接比较完成数量不同的总体均值。', '']
    for p in pairs:
        lines.append(f"- {p['comparison']}：{p['matched']} 组配对，ACC 差 {100*p['mean_acc_delta']:+.2f} 个百分点，"
                     f"胜/平/负 {p['wins']}/{p['ties']}/{p['losses']}。")
    lines += ['', '教师只用训练内部拟合，固定末轮生成收益；测试标签不参与收益目标。',
              '原选轮规则依赖测试 ACC，因此结果继续用于探索性比较。']
    (out/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def audit(out, args, data):
    for seed in args.seeds:
        for fold in args.folds:
            fit, test = [mask.cpu().numpy() for mask in data['Mask'][fold-1]]
            for mode in MODES:
                folder = folder_for(out, seed, mode, fold)
                r = read(folder/'complete.json')
                if mode != 'c_real' or not r.get('reused_baseline'):
                    rows = list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8')))
                    acc = [float(row['test_acc']) for row in rows]
                    assert len(rows) == args.epochs and np.argmax(acc)+1 == r['best']['epoch']
                assert r['last']['epoch'] == args.epochs
            cross = folder_for(out, seed, 'crossfit', fold)
            with np.load(cross/'targets.npz') as targets:
                np.testing.assert_array_equal(targets['fit_mask'], fit)
                np.testing.assert_array_equal(targets['test_mask'], test)
                assert np.isfinite(targets['gains'][fit]).all() and np.isnan(targets['gains'][test]).all()
                coverage = np.zeros(len(fit), dtype=int)
                for inner in range(2):
                    with np.load(cross/f'inner_{inner}'/'coalitions.npz') as coalitions:
                        train, held = coalitions['train_mask'], coalitions['held_mask']
                        assert not np.any(train & held) and not np.any((train | held) & test)
                        np.testing.assert_array_equal(train | held, fit)
                        coverage += held
                        np.testing.assert_allclose(targets['gains'][held], mean_marginal_gain(coalitions['losses'],
                                                   targets['gains'].shape[1]), rtol=0, atol=0)
                np.testing.assert_array_equal(coverage, fit.astype(int))
    dump(out/'audit.json', dict(passed=True, checked_at=timestamp(),
        checks=['all epochs and first TEST-ACC maxima', 'matched gate initialization/parameter count',
                'inner fitting/holdout disjointness', 'outer test targets all NaN', 'target reconstruction']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='Result/reference_selection')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--teacher-epochs', type=int)
    parser.add_argument('--gain-weight', type=float, default=.1)
    parser.add_argument('--gain-temperature', type=float, default=.05)
    parser.add_argument('--baseline-source', default='Result/relation_mechanism_nodecay')
    parser.add_argument('--config', default='Config/main_tadpole_3class.ini')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg, dd, data, graphs, paths = setup(args.cuda, args.config, args.k)
    args.epochs = args.epochs or cfg.epochs
    args.teacher_epochs = args.teacher_epochs or args.epochs
    args.modes, args.graph_weight_decay = list(MODES), 0.
    if args.epochs < 1 or args.teacher_epochs < 1 or args.gain_weight <= 0 or args.gain_temperature <= 0:
        raise ValueError('Invalid training budget or gain parameters')
    if any(f not in range(1, 11) for f in args.folds) or len(set(args.folds)) != len(args.folds):
        raise ValueError('Invalid folds')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('Invalid training seeds')
    if args.epochs != cfg.epochs and args.baseline_source:
        raise ValueError('Short smoke runs require --baseline-source "" to train a matched short baseline')
    out = (ROOT/args.output).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {out}; use --resume for incomplete folds')
    protocol = build_protocol(args, cfg, dd, data, graphs, paths)
    protocol.update(experiment='C: random neighbors, ordinary receive gate, cross-fitted gain-supervised receive gate',
        seeds=args.seeds, training_seed=None, teacher_epochs=args.teacher_epochs,
        teacher_inner_folds=2, teacher_selection='fixed last epoch; no held-out/test checkpoint selection',
        teacher_class_weights='inner fit label counts only; held/test labels replaced in training criterion',
        teacher_masks='per patient: 25% all off, 25% all on, 50% independent Bernoulli(.5)',
        gains='NLL(omit)-NLL(include), uniform average of 32 paired subsets for each of six modalities',
        gain_temperature=args.gain_temperature, gain_weight=args.gain_weight,
        gate='shared MLP(3H+8,32,1); clamp(1.2*sigmoid(logit)-.1,0,1); no sparsity loss',
        gate_inputs='detached own token, own mean token, normalized foreign contribution, learned modality ID',
        gate_scope='per recipient patient and CONTENT modality; off-diagonal blocks of composed two-pass operator',
        gate_limitation='own diagonal block remains graph dependent; gate off is not independently trained self-only',
        gate_initialization='both gated variants identical; all common C parameters copied exactly',
        random_graph='10 successful undirected double-edge swaps per edge, each node degree preserved; endpoint cosine recomputed',
        baseline_source=str((ROOT/args.baseline_source).resolve()) if args.baseline_source else None,
        parameter_matching='ordinary gate and gain gate identical; gates add parameters to C',
        mode_overrides={mode: NAMES[mode] for mode in MODES})
    for source in ('reference_selection_experiment.py', 'Model/reference_selection.py', 'Utils/reference_selection.py'):
        protocol['code_sha256'][source] = sha(ROOT/source)
    if (out/'PROTOCOL.json').exists() and read(out/'PROTOCOL.json') != protocol:
        raise ValueError('Code/protocol/data changed: use a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'PROTOCOL.json', protocol)
    np.savez_compressed(out/'splits_and_graphs.npz', source_row=np.asarray(dd['Index']),
        labels=data['Label'].cpu().numpy(), test_masks=np.stack([t.cpu().numpy() for _, t in data['Mask']]),
        **{f'edge_{m}': g['edge'].numpy() for m, g in enumerate(graphs)})
    try:
        import_baselines(out, args, cfg, dd, data, graphs, protocol)
        summary(out, args)
        for seed in args.seeds:
            status(out, state='running', stage='rewiring_candidates', seed=seed)
            random_graphs, stats = rewire_candidates(graphs, data['Feature'], dd['Modal_Index'], seed=7109+seed*100)
            dump(out/f'random_graph_seed_{seed}.json', stats)
            np.savez_compressed(out/f'random_graph_seed_{seed}.npz',
                **{f'edge_{m}': g['edge'].numpy() for m, g in enumerate(random_graphs)},
                **{f'similarity_{m}': g['similarity'].numpy() for m, g in enumerate(random_graphs)})
            for fold in args.folds:
                for mode in ('c_real', 'c_random', 'gate'):
                    outer_fold(out, args, cfg, dd, data, random_graphs if mode == 'c_random' else graphs, seed, fold, mode)
                    summary(out, args)
                gains = teacher_targets(out, args, cfg, dd, data, graphs, seed, fold)
                outer_fold(out, args, cfg, dd, data, graphs, seed, fold, 'gain_gate', gains)
                summary(out, args)
        audit(out, args, data)
        dump(out/'COMPLETE.json', summary(out, args))
        status(out, state='complete')
    except BaseException as error:
        status(out, state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
