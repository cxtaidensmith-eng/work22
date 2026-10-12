"""Bounded, frozen-network C parameter search on ABIDE-5 ADS/CN.

The historical first-maximum TEST ACC rule is retained. Screening folds are
reused test folds, not independent validation. Importing this module does not
load data, initialize CUDA, create artifacts or start training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import configparser
import copy
import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)

import diagnose_abide_capacity as runner
from alternating_cooperative_experiment import read, verify_manifest
from Loss import MSCQGNLoss
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import ROOT, dump, setup, sha, timestamp
from Utils import CustomCosineAnnealingLR, SET_Random
from Utils.relation_optimizer import (graph_only_parameter, optimizer_manifest,
                                      relation_optimizer)

OUT = ROOT/'Result/abide5_fine_tune_20260928'
REF = ROOT/'Result/c_abide5_seed0_20260924'
CONFIG = 'Config/c_abide5_transfer.ini'
SCREEN = (1, 3, 6, 8)
REMAINING = tuple(f for f in range(1, 11) if f not in SCREEN)
PARAMETERS = 768348
SAMPLES = 864
RANK_DECIMALS = 12
BASE = dict(lr=.006, weight_decay=.0007, Drop_rate=.64,
            input_noise_std=.1, aux_loss_weight=1., Loss_rate=.0001,
            graph_weight_decay=0.)
COARSE = {
    'L003': dict(lr=.003), 'L0015': dict(lr=.0015),
    'D050': dict(Drop_rate=.50), 'N005': dict(input_noise_std=.05),
    'N015': dict(input_noise_std=.15), 'W002': dict(weight_decay=.002),
    'A050': dict(aux_loss_weight=.50), 'O00001': dict(Loss_rate=.00001),
    'G005': dict(graph_weight_decay=.0005),
}
REFINEMENT_RULES = [
    dict(name='R_L080', key='lr', operation='multiply', value=.8),
    dict(name='R_L120', key='lr', operation='multiply', value=1.2),
    dict(name='R_DMINUS', key='Drop_rate', operation='add_clamp', value=-.08,
         lower=.2, upper=.8),
    dict(name='R_DPLUS', key='Drop_rate', operation='add_clamp', value=.08,
         lower=.2, upper=.8),
    dict(name='R_N075', key='input_noise_std', operation='multiply', value=.75),
    dict(name='R_N125', key='input_noise_std', operation='multiply', value=1.25),
]
METRIC_KEYS = ('acc', 'macro_f1', 'balanced_acc', 'ads_auc', 'ads_ap',
               'ads_recall', 'cn_recall')


def configure(cfg, spec):
    assert set(spec) == set(BASE)
    conf = copy.copy(cfg)
    for key, value in spec.items():
        assert math.isfinite(value) and value >= 0, (key, value)
        setattr(conf, key, value)
    assert 0 < conf.Drop_rate < 1 and conf.input_noise_std > 0
    assert conf.lr > conf.Lr_Min
    return conf


def retained_optimizer(model, cfg, graph_weight_decay=None):
    """Read this driver's explicit setting, not the legacy loop's .0005."""
    return relation_optimizer(model, cfg, cfg.graph_weight_decay)


def metrics(z, y):
    z, y = np.asarray(z), np.asarray(y)
    assert z.shape == (len(y), 2) and np.isfinite(z).all()
    assert set(np.unique(y)) == {0, 1}
    exp = np.exp(z-z.max(1, keepdims=True))
    probability = exp/exp.sum(1, keepdims=True)
    pred = z.argmax(1)
    recalls = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    ads = (y == 0).astype(int)
    return dict(acc=float((pred == y).mean()),
        macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y, pred)),
        ads_auc=float(roc_auc_score(ads, probability[:, 0])),
        ads_ap=float(average_precision_score(ads, probability[:, 0])),
        ads_recall=float(recalls[0]), cn_recall=float(recalls[1]),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def folder_for(name, fold):
    return (REF/'content' if name == 'BASE' else OUT/'runs'/name)/f'fold_{fold:02d}'


def collect(name, fold):
    folder = folder_for(name, fold)
    record = read(folder/'complete.json')
    with (folder/'epochs.csv').open(encoding='utf-8') as stream:
        curve = list(csv.DictReader(stream))
    assert [int(row['epoch']) for row in curve] == list(range(1, 1001))
    assert int(np.argmax([float(row['test_acc']) for row in curve]))+1 == record['best']['epoch']
    assert record['parameters'] == PARAMETERS
    result = dict(name=name, fold=fold, best_epoch=record['best']['epoch'],
        parameters=record['parameters'],
        late100_acc=float(np.mean([float(row['test_acc']) for row in curve[-100:]])))
    for stage, fn in [('best', 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'),
                      ('last', 'last_predictions.npz')]:
        with np.load(folder/fn) as saved:
            result[stage] = metrics(saved['logits'], saved['labels'])
        expected = record['best']['acc'] if stage == 'best' else float(curve[-1]['test_acc'])
        assert abs(result[stage]['acc']-expected) < 1e-12
    return result


def aggregate(name, folds):
    folds = tuple(folds)
    assert folds and len(folds) == len(set(folds))
    rows = [collect(name, fold) for fold in folds]
    result = {stage: {key: float(np.mean([row[stage][key] for row in rows]))
                      for key in METRIC_KEYS} for stage in ('best', 'last')}
    result.update(folds=list(folds),
        late100_acc=float(np.mean([row['late100_acc'] for row in rows])),
        acc_std=float(np.std([row['best']['acc'] for row in rows])))
    return result


def rank(names, folds):
    def key(name):
        result = aggregate(name, folds)
        return tuple(round(value, RANK_DECIMALS) for value in (
            result['best']['acc'], result['best']['macro_f1'],
            result['best']['balanced_acc'], result['late100_acc']))
    # Python's stable sort retains insertion order when all keys tie.
    return sorted(names, key=key, reverse=True)


def config_text(spec):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ROOT/CONFIG, encoding='utf-8')
    for key, value in spec.items():
        section = 'Modal' if key in ('Drop_rate', 'input_noise_std') else 'Optim'
        config.set(section, key, str(value))
    stream = io.StringIO()
    stream.write('# Reproduce with abide5_fine_tune.py: Config_ ignores graph_weight_decay.\n')
    config.write(stream)
    return stream.getvalue()


def write_config(name, spec):
    path = OUT/'configs'/f'{name}.ini'
    content = config_text(spec)
    if path.exists():
        assert path.read_text(encoding='utf-8') == content, path
    else:
        path.write_text(content, encoding='utf-8')
    manifest_path = OUT/'config_sha256.json'
    manifest = read(manifest_path) if manifest_path.exists() else {}
    for old_name, old_hash in manifest.items():
        assert sha(OUT/'configs'/f'{old_name}.ini') == old_hash
    value = sha(path)
    if name in manifest:
        assert manifest[name] == value
    else:
        manifest[name] = value
        dump(manifest_path, manifest)


def check_archive(path, dd, data, graphs):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        test_masks = torch.stack([test for _, test in data['Mask']]).cpu().numpy()
        np.testing.assert_array_equal(saved['test_masks'], test_masks)
        assert np.all(test_masks.sum(0) == 1)
        for index, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{index}'], graph['edge'].numpy())
            np.testing.assert_array_equal(saved[f'neighbors_{index}'], graph['neighbors'].numpy())
    for fit, test in data['Mask']:
        assert not (fit & test).any() and (fit | test).all()
    assert torch.isfinite(data['Feature']).all()


def prepare(cfg, dd, data, graphs, paths):
    previous = read(REF/'PLAN.json')
    verify_manifest(previous['code_sha256'])
    assert read(REF/'VERIFIED_COMPLETE.json')['audit']['passed']
    assert cfg.Device.type == 'cuda' and cfg.epochs == 1000 and cfg.seed == 0
    assert dd['Class_Names'] == ['ADS', 'CN'] and dd['Sample_Num'] == SAMPLES
    assert dd['Modal_Name'] == ['PHENO', 'ANAT', 'FUNC', 'MRI', 'fMRI']
    assert list(map(len, dd['Modal_Index'])) == [48, 6, 10, 150, 256]
    assert torch.bincount(data['Label']).cpu().tolist() == [397, 467]
    for key, value in previous['protocol']['original_config'].items():
        assert getattr(cfg, key) == value, key
    for key, value in BASE.items():
        if key != 'graph_weight_decay':
            assert getattr(cfg, key) == value, key
    for path in paths:
        assert sha(path) == previous['protocol']['data_sha256'][Path(path).name], path
    check_archive(REF/'splits_and_graphs.npz', dd, data, graphs)
    assert previous['protocol']['candidate_hash'] == hashlib.sha256(
        b''.join(graph['edge'].numpy().tobytes() for graph in graphs)).hexdigest()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        verify_manifest(plan['code_sha256'])
        verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
        assert plan['base'] == BASE and plan['coarse'] == COARSE
        assert plan['screen_folds'] == list(SCREEN)
        assert plan['refinement_rules'] == REFINEMENT_RULES
        check_archive(OUT/'data_splits_graphs.npz', dd, data, graphs)
        return plan
    (OUT/'configs').mkdir(exist_ok=True)
    (OUT/'runs').mkdir(exist_ok=True)
    sources = dict(previous['code_sha256'])
    for filename in ('abide5_fine_tune.py', 'diagnose_abide_capacity.py',
                     'alternating_cooperative_experiment.py', 'Utils/relation_optimizer.py',
                     'Utils/__init__.py', 'Loss/__init__.py'):
        sources[filename] = sha(ROOT/filename)
    refs = {}
    for filename in ('PLAN.json', 'PROTOCOL.json', 'audit.json', 'VERIFIED_COMPLETE.json',
                     'COMPLETE.json', 'splits_and_graphs.npz'):
        path = REF/filename
        refs[str(path.relative_to(ROOT))] = sha(path)
    for fold in range(1, 11):
        for filename in ('best.pt', 'predictions.npz', 'last_predictions.npz',
                         'epochs.csv', 'complete.json', 'optimizer_groups.json'):
            path = folder_for('BASE', fold)/filename
            refs[str(path.relative_to(ROOT))] = sha(path)
    shutil.copy2(REF/'splits_and_graphs.npz', OUT/'data_splits_graphs.npz')
    plan = dict(created_at=timestamp(), base=BASE, coarse=COARSE,
        refinement_rules=REFINEMENT_RULES, ranking_round_decimals=RANK_DECIMALS,
        seed=0, epochs=1000, screen_folds=list(SCREEN), remaining_folds=list(REMAINING),
        samples=SAMPLES, parameters=PARAMETERS, maximum_new_runs=150,
        expected_no_alias_minimum_new_runs=72,
        fixed='C content architecture; original five modalities, processed features, k10 graph support, '
              'tenfold splits, class weights, label smoothing .05, Adam, hold20/cosine1000/minlr.0001, '
              'clipping1, gate priors, depth5/width96, drop_path.1, seed0. No threshold or ensemble tuning.',
        selection='First maximum TEST ACC checkpoint. Nine single-factor candidates each screen on '
                  'folds1,3,6,8. Rank BASE and coarse candidates by mean best ACC, macroF1, balancedACC, '
                  'late100 ACC (each rounded12), stable insertion order. Around leader create six '
                  'predefined refinements, deduplicate exact full settings, and screen the same four folds. '
                  'Rank all nonBASE candidates with the same rule. Top2 plus every candidate whose '
                  'unrounded mean ACC is at least the second-ranked ACC minus1e-12 complete remaining6. '
                  'Final compare uniform complete-tenfold configurations against BASE using the same rank.',
        budget='actual_nonBASE_candidate_count*4 + finalist_count*6; maximum15*4+15*6=150. '
               'With no aliases and two finalists,72 new1000-epoch runs. No early stopping.',
        scope='Adaptive exploratory tuning with reused TEST folds and TEST checkpoint selection. '
              'Not independent validation, multiple-seed confirmation or proof of global optimum. '
              'ABIDE-5 results are not interchangeable with four-modality ABIDE.',
        graph_decay_reproduction='Optim.graph_weight_decay in exported INI is documentation only for '
              'the old Config_ loader. This driver configure()+retained_optimizer() explicitly applies it.',
        baseline_last_limitation='Historical last predictions are checked, but no last checkpoint exists.',
        baseline_protocol=previous['protocol'], code_sha256=sources, reference_sha256=refs,
        split_archive_sha256=sha(OUT/'data_splits_graphs.npz'), counts=[397, 467],
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(cfg.Device)))
    for filename in sources:
        target = OUT/'source_snapshot'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    dump(OUT/'PLAN.json', plan)
    return plan


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def verify_groups(model, opt, spec):
    groups = optimizer_manifest(model, opt)
    assert [group['weight_decay'] for group in groups] == [
        spec['weight_decay'], spec['graph_weight_decay']]
    assert all(group['lr'] == spec['lr'] for group in groups)
    assert groups[0]['parameters']+groups[1]['parameters'] == PARAMETERS
    for group in groups:
        assert all(graph_only_parameter(name) == (group['group'] == 1)
                   for name in group['names'])
    return groups


def preflight(cfg, dd, data, graphs, specs):
    """Incremental checks, always anchored to the original BASE initialization."""
    path = OUT/'preflight.json'
    result = read(path) if path.exists() else dict(passed=False, checks={}, baseline_checkpoints=[])
    for name, check in result['checks'].items():
        assert check['spec'] == specs[name] and check['passed']
    pending = [name for name in specs if name not in result['checks']]
    if not pending and len(result['baseline_checkpoints']) == 10:
        assert result['passed']
        return
    with (folder_for('BASE', 1)/'epochs.csv').open(encoding='utf-8') as stream:
        expected = list(csv.DictReader(stream))[:3]
    SET_Random(0)
    anchor = RelationSheafModel(dd, configure(cfg, BASE), graphs, 'content', training_seed=0).to(cfg.Device)
    assert sum(parameter.numel() for parameter in anchor.parameters()) == PARAMETERS
    initial_hash = parameter_digest(anchor)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(cfg.Device).clone()
    anchor.eval()
    with torch.no_grad():
        reference_logits = anchor(data['Feature'])[0].detach().clone()
    del anchor
    for name in pending:
        spec = specs[name]
        conf = configure(cfg, spec)
        SET_Random(0)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device)
        assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
        assert parameter_digest(model) == initial_hash
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, atol=0, rtol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(cfg.Device), cuda_rng, atol=0, rtol=0)
        effective_noise = model._modal_noise_std*model.noise_scale
        torch.testing.assert_close(effective_noise.cpu(), torch.full((5,), 3*spec['input_noise_std']))
        torch.testing.assert_close(model.modal_gate_logit.detach().cpu(), torch.tensor([-2., -2., -2., -1., -1.]))
        model.eval()
        with torch.no_grad():
            logits = model(data['Feature'])[0]
        torch.testing.assert_close(logits, reference_logits, atol=2e-7, rtol=1e-6)
        opt = retained_optimizer(model, conf)
        groups = verify_groups(model, opt, spec)
        criterion = MSCQGNLoss(dd, cfg.Device, rate=conf.Loss_rate, aux_loss_weight=conf.aux_loss_weight)
        assert criterion.CE_loss.label_smoothing == .05
        scheduler = CustomCosineAnnealingLR(opt, T_max=conf.T_max, eta_min=conf.Lr_Min)
        trace = []
        for epoch in range(1, 4):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(data['Feature'])
            loss = criterion(logits, data['Label'], data['Mask'][0][0], embeddings, auxiliary)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip, error_if_nonfinite=True)
            opt.step()
            _, detail = runner.assess(model, data, 1, criterion)
            if name == 'BASE':
                assert abs(float(loss.detach())-float(expected[epoch-1]['train_loss'])) < 2e-5
                assert abs(detail['test_acc']-float(expected[epoch-1]['test_acc'])) < 1e-12
            trace.append(dict(epoch=epoch, loss=float(loss.detach()),
                              test_acc=detail['test_acc'], lr=opt.param_groups[0]['lr']))
            scheduler.step()
        result['checks'][name] = dict(passed=True, spec=spec, parameters=PARAMETERS,
            first3=trace, init_parameter_sha256=initial_hash,
            cpu_rng_sha256=tensor_digest(cpu_rng), cuda_rng_sha256=tensor_digest(cuda_rng),
            effective_noise_sd=effective_noise.detach().cpu().tolist(), optimizer_groups=groups,
            initialization_and_eval_equal_BASE=True)
        del model, opt, criterion, scheduler, logits, embeddings, auxiliary, loss
        dump(path, result)
    del reference_logits
    if len(result['baseline_checkpoints']) != 10:
        checks = []
        conf = configure(cfg, BASE)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
        for fold in range(1, 11):
            folder = folder_for('BASE', fold)
            checkpoint = torch.load(folder/'best.pt', map_location=cfg.Device, weights_only=False)
            assert (checkpoint['mode'], checkpoint['fold'], checkpoint['seed']) == ('content', fold, 0)
            record = collect('BASE', fold)
            assert checkpoint['best']['epoch'] == record['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            test = data['Mask'][fold-1][1]
            with torch.no_grad():
                logits = model(data['Feature'])[0][test].cpu().numpy()
            with np.load(folder/'predictions.npz') as saved:
                np.testing.assert_allclose(logits, saved['logits'], atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(logits.argmax(1), saved['logits'].argmax(1))
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
            checks.append(dict(fold=fold, passed=True, best_epoch=record['best_epoch'],
                               checkpoint_sha256=sha(folder/'best.pt')))
        result['baseline_checkpoints'] = checks
        del model, logits
    result['passed'] = True
    result['verified_reference_checkpoints'] = 10
    result['updated_at'] = timestamp()
    dump(path, result)
    torch.cuda.empty_cache()


def audit_run(name, fold, spec, cfg, dd, data, graphs):
    folder = folder_for(name, fold)
    conf = configure(cfg, spec)
    record = collect(name, fold)
    groups = read(folder/'optimizer_groups.json')
    assert [group['weight_decay'] for group in groups] == [spec['weight_decay'], spec['graph_weight_decay']]
    assert all(group['lr'] == spec['lr'] for group in groups)
    assert sum(group['parameters'] for group in groups) == PARAMETERS
    for group in groups:
        assert all(graph_only_parameter(n) == (group['group'] == 1) for n in group['names'])
    with (folder/'epochs.csv').open(encoding='utf-8') as stream:
        curve = list(csv.DictReader(stream))
    assert all(np.isfinite(float(row[key])) for row in curve
               for key in row if key != 'epoch')
    for row in curve:
        step = int(row['epoch'])-1
        expected_lr = spec['lr'] if step < 20 else (
            conf.Lr_Min+(spec['lr']-conf.Lr_Min)*(1+math.cos(math.pi*(step-20)/980))/2)
        assert abs(float(row['lr'])-expected_lr) < 1e-12
    model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    expected_buffer = model._modal_noise_std.clone()
    test = data['Mask'][fold-1][1]
    row_index = np.asarray(dd['Index'])[test.cpu().numpy()]
    labels = data['Label'][test].cpu().numpy()
    checks = []
    for stage in ('best', 'last'):
        checkpoint = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
        assert (checkpoint['arm'], checkpoint['fold'], checkpoint['seed']) == (name, fold, 0)
        if stage == 'best':
            assert checkpoint['best']['epoch'] == record['best_epoch']
        else:
            assert checkpoint['epoch'] == 1000
        model.load_state_dict(checkpoint['state_dict'])
        torch.testing.assert_close(model._modal_noise_std, expected_buffer, atol=0, rtol=0)
        with torch.no_grad():
            logits = model(data['Feature'])[0][test].cpu().numpy()
        with np.load(folder/f'{stage}_predictions.npz') as saved:
            np.testing.assert_allclose(logits, saved['logits'], atol=2e-5, rtol=2e-5)
            np.testing.assert_array_equal(logits.argmax(1), saved['logits'].argmax(1))
            np.testing.assert_array_equal(labels, saved['labels'])
            if stage == 'best':
                np.testing.assert_array_equal(row_index, saved['row_index'])
        checks.append(dict(stage=stage, passed=True, sha256=sha(folder/f'{stage}.pt'),
                           metrics=metrics(logits, labels)))
    with np.load(folder_for('BASE', fold)/'predictions.npz') as saved:
        np.testing.assert_array_equal(row_index, saved['row_index'])
        np.testing.assert_array_equal(labels, saved['labels'])
    audit = dict(passed=True, name=name, fold=fold, settings=spec, parameters=PARAMETERS,
                 effective_noise_sd=(model._modal_noise_std*model.noise_scale).cpu().tolist(),
                 epoch_count=1000, first_best_epoch=record['best_epoch'], checks=checks)
    dump(folder/'fine_audit.json', audit)
    del model
    return audit


def report(specs):
    lines = ['# ABIDE-5：C 精细调参', '',
        'ADS/CN，864例，五模态。固定seed0、原十折、每折1000轮、首次最高测试ACC选轮。',
        'C网络、输入数据和图支持不变；四折筛选与未完成十折的结果不等同于完整十折成绩。', '',
        '| 配置 | 完成折数 | 已完成折最佳ACC | 宏F1 | ADS召回 | CN召回 | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    records = {}
    for name in specs:
        folds = [fold for fold in range(1, 11) if (folder_for(name, fold)/'complete.json').exists()]
        if not folds:
            continue
        result = aggregate(name, folds)
        records[name] = result
        lines.append(f"| {name} | {len(folds)}/10 | {100*result['best']['acc']:.4f}% | "
                     f"{100*result['best']['macro_f1']:.4f}% | {100*result['best']['ads_recall']:.2f}% | "
                     f"{100*result['best']['cn_recall']:.2f}% | {100*result['last']['acc']:.4f}% |")
    dump(OUT/'summary.json', dict(updated_at=timestamp(), candidates=records))
    lines += ['', 'input_noise_std是缩放配置；五组实际训练噪声标准差均为其3倍。',
              '图权重衰减由本实验驱动显式施加；旧入口不会自动读取导出INI中的该字段。',
              '所有候选与结果均保留；不按折混选、不调分类阈值。此搜索重复使用测试折，属于探索性结果。']
    (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_one(name, fold, specs, cfg, dd, data, graphs, stage):
    spec = specs[name]
    check = read(OUT/'preflight.json')['checks'][name]
    assert check['passed'] and check['spec'] == spec
    conf = configure(cfg, spec)
    dump(OUT/'status.json', dict(state=stage, arm=name, fold=fold, settings=spec, updated_at=timestamp()))
    folder = folder_for(name, fold)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'effective_settings.json').exists():
        assert read(folder/'effective_settings.json') == spec
    else:
        dump(folder/'effective_settings.json', spec)
    result = runner.run(name, fold, conf, dd, data, graphs)
    assert result['audit_passed'] and result['parameters'] == PARAMETERS
    audit_run(name, fold, spec, cfg, dd, data, graphs)
    report(specs)
    torch.cuda.empty_cache()


def build_refinements(specs):
    coarse_order = rank(list(specs), SCREEN)
    leader = coarse_order[0]
    candidates, aliases = {}, {}
    for rule in REFINEMENT_RULES:
        name, key = rule['name'], rule['key']
        value = specs[leader][key]
        if rule['operation'] == 'multiply':
            value *= rule['value']
        else:
            value = min(rule['upper'], max(rule['lower'], value+rule['value']))
        candidate = {**specs[leader], key: round(value, 12)}
        duplicate = next((old for old, spec in specs.items() if spec == candidate), None)
        if duplicate is not None:
            aliases[name] = duplicate
        else:
            specs[name] = candidate
            candidates[name] = candidate
    return dict(leader=leader, coarse_order=coarse_order, candidates=candidates, aliases=aliases)


def select_finalists(specs):
    order = rank([name for name in specs if name != 'BASE'], SCREEN)
    assert len(order) >= 2
    cutoff = aggregate(order[1], SCREEN)['best']['acc']
    finalists = [name for name in order
                 if name in order[:2] or aggregate(name, SCREEN)['best']['acc'] >= cutoff-1e-12]
    return dict(screen_order=rank(list(specs), SCREEN), candidate_order=order,
                cutoff_acc=cutoff, finalists=finalists, screen_folds=list(SCREEN),
                remaining_folds=list(REMAINING),
                expected_new_runs=(len(specs)-1)*len(SCREEN)+len(finalists)*len(REMAINING))


def finish(specs, selection, plan, dd):
    names = [name for name in specs if name == 'BASE' or name in selection['finalists']]
    full_order = rank(names, tuple(range(1, 11)))
    retained = full_order[0]
    full = {name: aggregate(name, range(1, 11)) for name in names}
    pairs, rows, screen_vs_remaining = {}, [], {}
    (OUT/'oof').mkdir(exist_ok=True)
    for name in names:
        all_y, all_z, all_id, all_fold = [], [], [], []
        corrected = harmed = 0
        wins = ties = losses = 0
        for fold in range(1, 11):
            filename = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
            with np.load(folder_for(name, fold)/filename) as saved, np.load(
                    folder_for('BASE', fold)/'predictions.npz') as baseline:
                np.testing.assert_array_equal(saved['labels'], baseline['labels'])
                np.testing.assert_array_equal(saved['row_index'], baseline['row_index'])
                correct = saved['logits'].argmax(1) == saved['labels']
                old = baseline['logits'].argmax(1) == baseline['labels']
                corrected += int((correct & ~old).sum())
                harmed += int((~correct & old).sum())
                delta = int(correct.sum())-int(old.sum())
                wins += int(delta > 0)
                ties += int(delta == 0)
                losses += int(delta < 0)
                all_y.extend(saved['labels'])
                all_z.extend(saved['logits'])
                all_id.extend(saved['row_index'])
                all_fold.extend([fold]*len(saved['labels']))
            rows.append(collect(name, fold))
        assert len(all_id) == len(set(all_id)) == SAMPLES == dd['Sample_Num']
        np.testing.assert_array_equal(sorted(all_id), sorted(dd['Index']))
        pairs[name] = dict(corrected=corrected, harmed=harmed, wins=wins, ties=ties,
                           losses=losses, pooled=metrics(all_z, all_y))
        np.savez_compressed(OUT/'oof'/f'{name}.npz', labels=all_y, logits=all_z,
                            row_index=all_id, fold=all_fold)
        if name == retained:
            shutil.copy2(OUT/'oof'/f'{name}.npz', OUT/'retained_oof_predictions.npz')
        screen_vs_remaining[name] = {
            'screen': aggregate(name, SCREEN), 'remaining': aggregate(name, REMAINING),
            'screen_delta_acc': aggregate(name, SCREEN)['best']['acc']-aggregate('BASE', SCREEN)['best']['acc'],
            'remaining_delta_acc': aggregate(name, REMAINING)['best']['acc']-aggregate('BASE', REMAINING)['best']['acc']}
    new_runs = list((OUT/'runs').glob('*/fold_*/complete.json'))
    assert len(new_runs) == selection['expected_new_runs'] <= plan['maximum_new_runs']
    expected_pairs = {(name, fold) for name in specs if name != 'BASE' for fold in SCREEN}
    expected_pairs |= {(name, fold) for name in selection['finalists'] for fold in REMAINING}
    actual_pairs = {(path.parent.parent.name, int(path.parent.name.split('_')[1])) for path in new_runs}
    assert actual_pairs == expected_pairs
    assert all(read(path)['audit_passed'] and read(path.parent/'fine_audit.json')['passed'] for path in new_runs)
    for name, spec in specs.items():
        write_config(name, spec)
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
    result = dict(retained=retained, full_order=full_order, full=full, pairs=pairs,
        records=rows, settings=specs[retained], selection=selection,
        new_runs=len(new_runs), verified_new_checkpoints=2*len(new_runs),
        screen_vs_remaining=screen_vs_remaining)
    dump(OUT/'analysis.json', result)
    dump(OUT/'screen_vs_remaining.json', screen_vs_remaining)
    dump(OUT/'best_configuration.json', dict(name=retained,
        driver='abide5_fine_tune.py', base_config=CONFIG,
        config=str(OUT/'configs'/f'{retained}.ini'), overrides=specs[retained],
        model='RelationSheafModel(mode=content)', graph_weight_decay=specs[retained]['graph_weight_decay'],
        k=10, seed=0, epochs=1000, parameters=PARAMETERS,
        effective_noise_sd=[3*specs[retained]['input_noise_std']]*5,
        reproduction='Use configure(cfg, overrides) and retained_optimizer(model, cfg). '
                     'The old Config_ ignores the INI graph_weight_decay field; INI alone is insufficient.',
        selection='First maximum TEST ACC; best among this bounded search, not global optimum.'))
    assert read(OUT/'config_sha256.json') == {
        name: sha(OUT/'configs'/f'{name}.ini') for name in specs}
    report(specs)
    lines = ['', '## 完整十折比较', '',
        '| 配置 | 最佳ACC | 宏F1 | 平衡ACC | ADS召回 | CN召回 | ADS概率AUC | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in full.items():
        values = [result['best'][key] for key in
                  ('acc', 'macro_f1', 'balanced_acc', 'ads_recall', 'cn_recall', 'ads_auc')]
        values.append(result['last']['acc'])
        lines.append('| '+name+' | '+' | '.join(f'{100*value:.4f}%' for value in values)+' |')
    pair = pairs[retained]
    lines += ['', f'本轮保留：{retained}。相对BASE纠正{pair["corrected"]}例、新增{pair["harmed"]}例错误。',
        f'新增{len(new_runs)}次完整训练、{2*len(new_runs)}个新最佳/末轮检查点复核通过。',
        '每个完整配置覆盖864条数据行，各有一次折外预测；数据行唯一性不等同于临床患者ID独立性。',
        '原十折BASE的10个最佳检查点已重放；历史末轮仅有预测文件。',
        '筛选四折与剩余六折收益分别保存在screen_vs_remaining.json；所有折均沿用测试ACC选轮，后六折也不是独立最终验证集。',
        'ABIDE-5与ABIDE为不同特征/样本版本，不能混用成绩。当前结果仅对应seed0、原协议和已完成搜索范围。',
        '', '## 保留配置逐折结果', '', '| 折 | 最佳轮次 | 最佳ACC | BASE ACC | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|']
    for fold in range(1, 11):
        row = collect(retained, fold)
        old = collect('BASE', fold)
        lines.append(f"| {fold} | {row['best_epoch']} | {100*row['best']['acc']:.4f}% | "
                     f"{100*old['best']['acc']:.4f}% | {100*row['last']['acc']:.4f}% |")
    with (OUT/'进度与结果.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), all_audits_passed=True,
         new_runs=len(new_runs), verified_new_checkpoints=2*len(new_runs), retained=retained))
    dump(OUT/'status.json', dict(state='complete', retained=retained,
                               new_runs=len(new_runs), updated_at=timestamp()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true',
                        help='Freeze inputs and run preflight; do not start candidate1000-epoch training.')
    args = parser.parse_args()
    cfg, dd, data, graphs, paths = setup(0, CONFIG, 10)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        plan = prepare(cfg, dd, data, graphs, paths)
        if (OUT/'COMPLETE.json').exists():
            return
        specs = {'BASE': dict(BASE), **{name: {**BASE, **change} for name, change in COARSE.items()}}
        for name, spec in specs.items():
            write_config(name, spec)
        if not (OUT/'all_settings.json').exists():
            dump(OUT/'all_settings.json', specs)
        else:
            saved = read(OUT/'all_settings.json')
            for name, spec in specs.items():
                assert saved[name] == spec
            # A resumed refinement-stage run must retain previous preflight entries.
            if (OUT/'refinement.json').exists():
                old_refinement = read(OUT/'refinement.json')
                specs.update(old_refinement['candidates'])
                assert all(name in specs and spec == specs[name] for name, spec in saved.items())
                for name, spec in specs.items():
                    write_config(name, spec)
                dump(OUT/'all_settings.json', specs)
        runner.OUT = OUT/'runs'
        runner.RelationSheafModel = RelationSheafModel
        runner.relation_optimizer = retained_optimizer
        dump(OUT/'status.json', dict(state='preflight', updated_at=timestamp()))
        preflight(cfg, dd, data, graphs, specs)
        if args.prepare_only:
            report(specs)
            dump(OUT/'status.json', dict(state='prepared', preflight_passed=True, updated_at=timestamp()))
            print(json.dumps(dict(preflight=True, output=str(OUT), maximum_new_runs=150)), flush=True)
            return
        for name in COARSE:
            for fold in SCREEN:
                run_one(name, fold, specs, cfg, dd, data, graphs, 'coarse')
        # Derive deterministically from only BASE and the completed coarse candidates.
        coarse_specs = {name: specs[name] for name in ('BASE', *COARSE)}
        refinement = build_refinements(coarse_specs)
        specs = coarse_specs
        if (OUT/'refinement.json').exists():
            assert read(OUT/'refinement.json') == refinement
        else:
            dump(OUT/'refinement.json', refinement)
        for name, spec in specs.items():
            write_config(name, spec)
        dump(OUT/'all_settings.json', specs)
        dump(OUT/'status.json', dict(state='refinement_preflight', leader=refinement['leader'], updated_at=timestamp()))
        preflight(cfg, dd, data, graphs, specs)
        for name in refinement['candidates']:
            for fold in SCREEN:
                run_one(name, fold, specs, cfg, dd, data, graphs, 'refine')
        selection = select_finalists(specs)
        if (OUT/'selection.json').exists():
            assert read(OUT/'selection.json') == selection
        else:
            dump(OUT/'selection.json', selection)
        for name in selection['finalists']:
            for fold in REMAINING:
                run_one(name, fold, specs, cfg, dd, data, graphs, 'ten_fold')
        dump(OUT/'status.json', dict(state='finalizing', updated_at=timestamp()))
        finish(specs, selection, plan, dd)
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
