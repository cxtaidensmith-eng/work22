"""Bounded parameter search for frozen C on TADPOLE SMCI/PMCI, seed0.

Keeps the user's historical test-ACC checkpoint selection. Search-fold results
are exploratory and are never presented as independent validation estimates.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import copy
import csv
import configparser
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score, recall_score)
import diagnose_abide_capacity as runner
from relation_experiment import ROOT, setup, dump, sha, timestamp
from alternating_cooperative_experiment import read, verify_manifest
from Model.relation_sheaf import RelationSheafModel
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer
from Loss import MSCQGNLoss

OUT = ROOT/'Result/tad_binary_fine_tune_20260928'
REF = ROOT/'Result/c_tadpole_binary_seed0_20260924'
CONFIG = 'Config/c_tadpole_binary_transfer.ini'
SCREEN = (1, 6)
CHECK = (3, 8)
BASE = dict(lr=.006, weight_decay=.0007, Drop_rate=.64,
            input_noise_std=.1, aux_loss_weight=1., Loss_rate=.0001)
COARSE = {
    'L003': dict(lr=.003), 'L0015': dict(lr=.0015),
    'N005': dict(input_noise_std=.05), 'N015': dict(input_noise_std=.15),
    'D050': dict(Drop_rate=.50), 'A025': dict(aux_loss_weight=.25),
    'O00001': dict(Loss_rate=.00001), 'W002': dict(weight_decay=.002),
}


def configure(cfg, spec):
    conf = copy.copy(cfg)
    for key, value in spec.items():
        setattr(conf, key, value)
    return conf


def retained_optimizer(model, cfg, graph_weight_decay=None):
    # The reusable loop's old default is overridden explicitly for TAD C.
    return relation_optimizer(model, cfg, 0.)


def metrics(z, y):
    z, y = np.asarray(z), np.asarray(y)
    ex = np.exp(z-z.max(1, keepdims=True)); prob = ex/ex.sum(1, keepdims=True)
    pred = z.argmax(1)
    rec = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return dict(acc=float((pred == y).mean()),
        macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, pred)),
        pmci_auc=float(roc_auc_score(y, prob[:, 1])),
        pmci_ap=float(average_precision_score(y, prob[:, 1])),
        smci_recall=float(rec[0]), pmci_recall=float(rec[1]),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def folder_for(name, fold):
    return (REF/'content' if name == 'BASE' else OUT/'runs'/name)/f'fold_{fold:02d}'


def collect(name, fold):
    folder = folder_for(name, fold)
    record = read(folder/'complete.json')
    with (folder/'epochs.csv').open() as stream:
        curve = list(csv.DictReader(stream))
    assert [int(r['epoch']) for r in curve] == list(range(1, 1001))
    assert int(np.argmax([float(r['test_acc']) for r in curve]))+1 == record['best']['epoch']
    result = dict(name=name, fold=fold, best_epoch=record['best']['epoch'],
        parameters=record['parameters'],
        late100_acc=float(np.mean([float(r['test_acc']) for r in curve[-100:]])))
    filename = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
    for stage, fn in [('best', filename), ('last', 'last_predictions.npz')]:
        with np.load(folder/fn) as p:
            result[stage] = metrics(p['logits'], p['labels'])
        if stage == 'best':
            assert abs(result[stage]['acc']-record['best']['acc']) < 1e-12
    return result


def aggregate(name, folds):
    rows = [collect(name, f) for f in folds]
    keys = ('acc', 'macro_f1', 'balanced_acc', 'pmci_auc', 'pmci_ap', 'pmci_recall', 'smci_recall')
    result = {stage: {key: float(np.mean([r[stage][key] for r in rows])) for key in keys}
              for stage in ('best', 'last')}
    result.update(folds=list(folds), late100_acc=float(np.mean([r['late100_acc'] for r in rows])),
                  acc_std=float(np.std([r['best']['acc'] for r in rows])))
    return result


def rank(names, folds):
    def key(name):
        a = aggregate(name, folds)
        return (a['best']['acc'], a['best']['macro_f1'], a['best']['pmci_recall'], a['late100_acc'])
    return sorted(names, key=key, reverse=True)


def write_config(name, spec):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ROOT/CONFIG, encoding='utf-8')
    for key, value in spec.items():
        section = 'Modal' if key in ('Drop_rate', 'input_noise_std') else 'Optim'
        config.set(section, key, str(value))
    with (OUT/'configs'/f'{name}.ini').open('w', encoding='utf-8') as stream:
        config.write(stream)


def prepare(cfg, dd, data, graphs, paths):
    previous = read(REF/'PLAN.json')
    verify_manifest(previous['code_sha256'])
    for path in paths:
        assert sha(path) == previous['protocol']['data_sha256'][Path(path).name], path
    assert read(REF/'VERIFIED_COMPLETE.json')['audit']['passed']
    assert cfg.Device.type == 'cuda' and cfg.epochs == 1000 and cfg.seed == 0
    assert dd['Class_Names'] == ['SMCI', 'PMCI'] and dd['Sample_Num'] == 535
    for key, value in BASE.items():
        assert getattr(cfg, key) == value, key
    with np.load(REF/'splits_and_graphs.npz') as p:
        np.testing.assert_array_equal(p['source_row'], dd['Index'])
        np.testing.assert_array_equal(p['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(p['test_masks'], torch.stack([b for _, b in data['Mask']]).cpu().numpy())
    assert previous['protocol']['candidate_hash'] == __import__('hashlib').sha256(
        b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
        return plan
    OUT.mkdir(exist_ok=False); (OUT/'configs').mkdir(); (OUT/'runs').mkdir()
    sources = dict(previous['code_sha256'])
    for filename in ('tad_binary_fine_tune.py', 'diagnose_abide_capacity.py',
                     'alternating_cooperative_experiment.py', 'Utils/relation_optimizer.py'):
        sources[filename] = sha(ROOT/filename)
    refs = {}
    for f in range(1, 11):
        for filename in ('best.pt', 'predictions.npz', 'last_predictions.npz', 'epochs.csv', 'complete.json'):
            p = folder_for('BASE', f)/filename; refs[str(p.relative_to(ROOT))] = sha(p)
    shutil.copy2(REF/'splits_and_graphs.npz', OUT/'data_splits_graphs.npz')
    plan = dict(created_at=timestamp(), base=BASE, coarse=COARSE, seed=0, epochs=1000,
        screen_folds=SCREEN, check_folds=CHECK, maximum_new_runs=40,
        fixed='C content architecture and 716396 parameters; original five modalities, processed data, graphs k10, splits, class weights, label smoothing .05, Adam, cosine1000/minlr.0001, graph decay0, clipping1, gate priors, depth5/width96, no threshold tuning.',
        selection='Checkpoint remains first highest test ACC. Candidate ranking: mean best ACC, macroF1, PMCI recall, last100 mean ACC, stable insertion order. Screen eight one-factor settings on folds1,6. Select leader including BASE, create four local settings: lr*0.75, lr*1.25, auxiliary weight*0.5, orthogonality rate*0.1. Deduplicate exact settings. Top2 nonbaseline settings after screening get folds3,8 and then both complete remaining six folds. Final compare complete tenfold candidates against BASE with the same ranking; one configuration across all folds.',
        scope='Bounded exploratory tuning under retained TEST-selected protocol; screening/check folds are reused test folds, not independent validation. No global-optimum claim, architecture change, multiple seeds or per-fold configuration ensemble.',
        baseline_protocol=previous['protocol'], code_sha256=sources, reference_sha256=refs,
        split_archive_sha256=sha(OUT/'data_splits_graphs.npz'),
        counts=torch.bincount(data['Label']).cpu().tolist(),
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    for filename in sources:
        target = OUT/'source_snapshot'/filename; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    dump(OUT/'PLAN.json', plan)
    return plan


def preflight(cfg, dd, data, graphs, specs):
    if (OUT/'preflight.json').exists():
        return
    with (folder_for('BASE', 1)/'epochs.csv').open() as stream:
        expected = list(csv.DictReader(stream))[:3]
    checks = {}; init = None; eval_ref = None; rng = None
    for name, spec in specs.items():
        conf = configure(cfg, spec); SET_Random(0)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device)
        assert sum(p.numel() for p in model.parameters()) == 716396
        if init is None:
            init = {n: p.detach().clone() for n, p in model.named_parameters()}; rng = torch.get_rng_state()
        else:
            assert torch.equal(rng, torch.get_rng_state())
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, init[n], atol=0, rtol=0)
        model.eval()
        with torch.no_grad(): z = model(data['Feature'])[0]
        if eval_ref is None: eval_ref = z.clone()
        else: torch.testing.assert_close(z, eval_ref, atol=2e-7, rtol=1e-6)
        opt = retained_optimizer(model, conf)
        assert [g['weight_decay'] for g in opt.param_groups] == [spec['weight_decay'], 0.]
        criterion = MSCQGNLoss(dd, cfg.Device, rate=conf.Loss_rate, aux_loss_weight=conf.aux_loss_weight)
        sched = CustomCosineAnnealingLR(opt, T_max=conf.T_max, eta_min=conf.Lr_Min)
        losses = []
        for i in range(3):
            model.train(); opt.zero_grad(set_to_none=True)
            z, emb, aux = model(data['Feature']); loss = criterion(z, data['Label'], data['Mask'][0][0], emb, aux)
            assert torch.isfinite(loss)
            if name == 'BASE': assert abs(float(loss.detach())-float(expected[i]['train_loss'])) < 2e-5
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip, error_if_nonfinite=True); opt.step()
            _, detail = runner.assess(model, data, 1, criterion)
            if name == 'BASE': assert abs(detail['test_acc']-float(expected[i]['test_acc'])) < 1e-12
            sched.step(); losses.append(float(loss.detach()))
        checks[name] = dict(passed=True, first3_losses=losses,
            effective_noise_sd=(model._modal_noise_std*model.noise_scale).cpu().tolist())
        del model, opt, criterion, z, emb, aux, loss
    del init, eval_ref
    # All existing best checkpoints are replayed before their results are reused.
    model = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    for fold in range(1, 11):
        cp = torch.load(folder_for('BASE', fold)/'best.pt', map_location=cfg.Device, weights_only=False)
        model.load_state_dict(cp['state_dict'])
        test = data['Mask'][fold-1][1]
        with torch.no_grad(): z = model(data['Feature'])[0][test].cpu().numpy()
        with np.load(folder_for('BASE', fold)/'predictions.npz') as p:
            np.testing.assert_allclose(z, p['logits'], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(z.argmax(1), p['logits'].argmax(1))
            np.testing.assert_array_equal(p['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
            np.testing.assert_array_equal(p['labels'], data['Label'][test].cpu().numpy())
    dump(OUT/'preflight.json', dict(passed=True, checks=checks, verified_reference_checkpoints=10))
    del model, z; torch.cuda.empty_cache()


def report(specs):
    lines = ['# TAD 二分类：C 参数搜索', '',
        'SMCI/PMCI；固定seed0、原十折、每折1000轮，首次最高测试ACC选轮。网络结构、候选图与数据不变。',
        '两折/四折成绩仅用于探索筛选，不是完整十折或独立验证。', '',
        '| 配置 | 完成折数 | 已完成折最佳ACC | 宏F1 | PMCI召回 | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|']
    records = {}
    for name in specs:
        folds = [f for f in range(1, 11) if (folder_for(name, f)/'complete.json').exists()]
        if not folds: continue
        a = aggregate(name, folds); records[name] = a
        lines.append(f"| {name} | {len(folds)}/10 | {100*a['best']['acc']:.4f}% | {100*a['best']['macro_f1']:.4f}% | {100*a['best']['pmci_recall']:.2f}% | {100*a['last']['acc']:.4f}% |")
    dump(OUT/'summary.json', dict(updated_at=timestamp(), candidates=records))
    (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_one(name, fold, specs, cfg, dd, data, graphs, stage):
    spec = specs[name]; conf = configure(cfg, spec)
    dump(OUT/'status.json', dict(state=stage, arm=name, fold=fold, settings=spec, updated_at=timestamp()))
    folder = folder_for(name, fold); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'effective_settings.json').exists(): assert read(folder/'effective_settings.json') == spec
    else: dump(folder/'effective_settings.json', spec)
    result = runner.run(name, fold, conf, dd, data, graphs)
    assert result['audit_passed'] and result['parameters'] == 716396
    groups = read(folder/'optimizer_groups.json')
    assert [g['weight_decay'] for g in groups] == [spec['weight_decay'], 0.]
    with np.load(folder/'best_predictions.npz') as p, np.load(folder_for('BASE', fold)/'predictions.npz') as base:
        np.testing.assert_array_equal(p['row_index'], base['row_index'])
        np.testing.assert_array_equal(p['labels'], base['labels'])
    with np.load(folder/'last_predictions.npz') as p:
        np.testing.assert_array_equal(p['labels'], data['Label'][data['Mask'][fold-1][1]].cpu().numpy())
    report(specs); torch.cuda.empty_cache()


def finish(specs, finalists, plan, dd):
    names = ['BASE']+finalists
    retained = rank(names, tuple(range(1, 11)))[0]
    full = {name: aggregate(name, tuple(range(1, 11))) for name in names}
    pairs = {}; rows = []
    for name in names:
        all_y = []; all_z = []; all_id = []; corrected = harmed = 0
        for fold in range(1, 11):
            folder = folder_for(name, fold)
            fn = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
            with np.load(folder/fn) as p, np.load(folder_for('BASE', fold)/'predictions.npz') as base:
                np.testing.assert_array_equal(p['labels'], base['labels'])
                np.testing.assert_array_equal(p['row_index'], base['row_index'])
                correct = p['logits'].argmax(1) == p['labels']; old = base['logits'].argmax(1) == base['labels']
                corrected += int((correct & ~old).sum()); harmed += int((~correct & old).sum())
                all_y.extend(p['labels']); all_z.extend(p['logits']); all_id.extend(p['row_index'])
            rows.append(collect(name, fold))
        assert len(all_id) == len(set(all_id)) == dd['Sample_Num']
        np.testing.assert_array_equal(sorted(all_id), sorted(dd['Index']))
        pairs[name] = dict(corrected=corrected, harmed=harmed, pooled=metrics(all_z, all_y))
        if name == retained:
            np.savez_compressed(OUT/'retained_oof_predictions.npz', labels=all_y, logits=all_z, row_index=all_id)
    new_runs = list((OUT/'runs').glob('*/fold_*/complete.json'))
    assert len(new_runs) <= plan['maximum_new_runs']
    assert all(read(p)['audit_passed'] for p in new_runs)
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
    result = dict(retained=retained, full=full, pairs=pairs, records=rows, settings=specs[retained],
                  new_runs=len(new_runs), verified_new_checkpoints=2*len(new_runs))
    dump(OUT/'analysis.json', result)
    dump(OUT/'best_configuration.json', dict(name=retained, config=str(OUT/'configs'/f'{retained}.ini'),
        overrides=specs[retained], model='RelationSheafModel(mode=content)', graph_weight_decay=0., k=10,
        seed=0, epochs=1000, parameters=716396,
        selection='First maximum TEST ACC per fold; exploratory best within this bounded search, not global optimum.'))
    report(specs)
    lines = ['', '## 完整十折比较', '', '| 配置 | 最佳ACC | 宏F1 | PMCI召回 | PMCI概率AUC | 第1000轮ACC |',
             '|---|---:|---:|---:|---:|---:|']
    for name, a in full.items():
        lines.append(f"| {name} | {100*a['best']['acc']:.4f}% | {100*a['best']['macro_f1']:.4f}% | {100*a['best']['pmci_recall']:.2f}% | {100*a['best']['pmci_auc']:.4f}% | {100*a['last']['acc']:.4f}% |")
    lines += ['', f'本轮保留：{retained}。新增{len(new_runs)}次完整训练，{2*len(new_runs)}个新最佳/末轮检查点复核通过。',
              '每个完整配置覆盖535条数据行，各有一次折外预测。行号核查不等同于临床患者ID独立性核查。',
              '所有结果仍为原测试选轮与自适应测试折筛选下的探索性结果，不能解释为独立泛化确认，也不证明全局最高性能。']
    with (OUT/'进度与结果.md').open('a', encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), all_audits_passed=True,
         new_runs=len(new_runs), retained=retained))
    dump(OUT/'status.json', dict(state='complete', retained=retained, new_runs=len(new_runs), updated_at=timestamp()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    cfg, dd, data, graphs, paths = setup(0, CONFIG, 10)
    plan = prepare(cfg, dd, data, graphs, paths)
    specs = {'BASE': dict(BASE), **{name: {**BASE, **change} for name, change in COARSE.items()}}
    for name, spec in specs.items(): write_config(name, spec)
    runner.OUT = OUT/'runs'; runner.relation_optimizer = retained_optimizer
    preflight(cfg, dd, data, graphs, specs)
    if args.prepare_only:
        report(specs); print(json.dumps(dict(preflight=True, output=str(OUT), max_runs=40)), flush=True); return
    if (OUT/'COMPLETE.json').exists(): return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        for name in COARSE:
            for fold in SCREEN: run_one(name, fold, specs, cfg, dd, data, graphs, 'coarse')
        leader = rank(list(specs), SCREEN)[0]
        refinements = [('R_L075', 'lr', .75), ('R_L125', 'lr', 1.25),
                       ('R_A050', 'aux_loss_weight', .5), ('R_O010', 'Loss_rate', .1)]
        aliases = {}
        for name, key, scale in refinements:
            candidate = {**specs[leader], key: round(specs[leader][key]*scale, 12)}
            duplicate = next((n for n, spec in specs.items() if spec == candidate), None)
            if duplicate:
                aliases[name] = duplicate; continue
            specs[name] = candidate; write_config(name, candidate)
        dump(OUT/'all_settings.json', specs)
        dump(OUT/'refinement.json', dict(leader=leader, aliases=aliases))
        for name in specs:
            if name == 'BASE' or name in COARSE: continue
            for fold in SCREEN: run_one(name, fold, specs, cfg, dd, data, graphs, 'refine')
        finalists = rank([n for n in specs if n != 'BASE'], SCREEN)[:2]
        dump(OUT/'selection.json', dict(screen_order=rank(list(specs), SCREEN), finalists=finalists))
        for name in finalists:
            for fold in CHECK: run_one(name, fold, specs, cfg, dd, data, graphs, 'four_fold')
        for name in rank(finalists, SCREEN+CHECK):
            for fold in range(1, 11):
                if fold not in SCREEN+CHECK: run_one(name, fold, specs, cfg, dd, data, graphs, 'ten_fold')
        finish(specs, finalists, plan, dd)
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp())); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
