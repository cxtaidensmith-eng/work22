"""Optional third TAD binary search, gated on a completed independent round-two audit.

Keep the audited C network, data, seed0, ten folds, 1000 epochs and first maximum
TEST-ACC checkpoint rule. Anchor every setting to the single round-two tenfold
winner. Search smoothing and auxiliary loss weight one factor at a time.
This file only prepares a future option; importing it never starts training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import configparser
import copy
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

import tad_binary_fine_tune as common
from alternating_cooperative_experiment import read, verify_manifest
from Loss import MSCQGNLoss as OriginalLoss
from Model.relation_sheaf import RelationSheafModel
from relation_experiment import ROOT, dump, setup, sha, timestamp
from Utils import CustomCosineAnnealingLR, SET_Random


OUT = ROOT/'Result/tad_binary_loss_tune_20260928'
ROUND2 = ROOT/'Result/tad_binary_joint_tune_20260928'
ROUND1 = ROOT/'Result/tad_binary_fine_tune_20260928'
ORIGINAL = ROOT/'Result/c_tadpole_binary_seed0_20260924'
CONFIG = 'Config/c_tadpole_binary_transfer.ini'
SCREEN = (1, 3, 6, 8)
REMAINING = (2, 4, 5, 7, 9, 10)
ALL_FOLDS = tuple(range(1, 11))
REPRODUCTION_ENTRYPOINT = 'tad_binary_loss_tune.py'
REPRODUCTION_NOTE = ('Legacy Config_ and training entry points do not read label_smoothing from the exported INI. '
    'Reproduce with this driver and its explicit loss_factory, or explicitly pass the selected label_smoothing '
    'to MSCQGNLoss for the main and both auxiliary CE losses; the INI alone is insufficient.')
TRAINING_KEYS = ('lr', 'weight_decay', 'Drop_rate', 'input_noise_std',
                 'aux_loss_weight', 'Loss_rate')
CHANGES = {
    'LS000': dict(label_smoothing=0.),
    'LS002': dict(label_smoothing=.02),
    'A050': dict(aux_loss_weight=.50),
    'A075': dict(aux_loss_weight=.75),
}
ANCHOR_INFO = None
ACTIVE_SMOOTHING = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def loss_factory(dataset, device, rate=.001, label_smoothing=None,
                 aux_loss_weight=1., aux_supervision='ovr'):
    """Explicit per-run smoothing for the unchanged shared training loop."""
    require(ACTIVE_SMOOTHING is not None, 'No active loss setting')
    if label_smoothing is not None:
        require(float(label_smoothing) == ACTIVE_SMOOTHING, 'Conflicting smoothing arguments')
    criterion = OriginalLoss(dataset, device, rate=rate,
        label_smoothing=ACTIVE_SMOOTHING, aux_loss_weight=aux_loss_weight,
        aux_supervision=aux_supervision)
    require(criterion.CE_loss.label_smoothing == ACTIVE_SMOOTHING, 'Main smoothing was not applied')
    require(len(criterion.aux_loss_dict) == 2, 'Expected two binary OVR heads')
    require(all(module.label_smoothing == ACTIVE_SMOOTHING
                for module in criterion.aux_loss_dict.values()), 'Auxiliary smoothing was not applied')
    return criterion


def activate_loss(value):
    global ACTIVE_SMOOTHING
    require(math.isfinite(float(value)) and 0 <= value < 1, 'Invalid label smoothing')
    ACTIVE_SMOOTHING = float(value)


def prerequisite():
    """Run before setup so an unfinished prior round never allocates the GPU."""
    require((ROUND2/'COMPLETE.json').is_file(), 'Round two must finish before preparing or training round three')
    require(not (ROUND2/'RUNNING.lock').exists(), 'Round two still has a training lock')
    require((ROUND2/'independent_audit.json').is_file(), 'Round two independent audit is required')
    completion, audit, analysis, exported, plan = (read(ROUND2/name) for name in
        ('COMPLETE.json', 'independent_audit.json', 'analysis.json', 'best_configuration.json', 'PLAN.json'))
    require(completion['all_audits_passed'] is True and audit['passed'] is True,
            'Round two completion and independent audit must both pass')
    name = analysis['retained']
    require(completion['retained'] == audit['retained'] == exported['name'] == name,
            'Round two winner identity differs between artifacts')
    require(exported['overrides'] == analysis['settings'] == read(ROUND2/'all_settings.json')[name],
            'Round two winner settings disagree')
    require(name in analysis['full'] and analysis['full'][name]['folds'] == list(ALL_FOLDS),
            'Anchor must be a complete tenfold configuration')
    require(set(exported['overrides']) == set(TRAINING_KEYS), 'Unexpected anchor hyperparameters')
    for key, expected in dict(seed=0, epochs=1000, parameters=716396, k=10,
                              graph_weight_decay=0., model='RelationSheafModel(mode=content)').items():
        require(exported[key] == expected, f'Anchor protocol changed:{key}')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    for filename, digest in audit['artifact_sha256'].items():
        require(sha(ROUND2/filename) == digest, f'Round two audited artifact changed:{filename}')
    require(sha(ROOT/'audit_tad_binary_joint_tune.py') == audit['auditor_sha256'],
            'Round two independent auditor source changed after verification')
    require(sha(ROOT/'audit_tad_binary_fine_tune.py') == audit['shared_independent_auditor_sha256'],
            'Shared independent auditor source changed after verification')
    require(sha(ROUND2/'data_splits_graphs.npz') == plan['split_archive_sha256'], 'Round two split archive changed')
    folders = {}
    for fold in ALL_FOLDS:
        if name == 'BASE':
            folder = ORIGINAL/'content'/f'fold_{fold:02d}'
        elif fold in plan['reused_folds'].get(name, []):
            folder = ROUND1/'runs'/name/f'fold_{fold:02d}'
        else:
            folder = ROUND2/'runs'/name/f'fold_{fold:02d}'
        require((folder/'complete.json').is_file(), f'Missing anchor fold:{folder}')
        folders[str(fold)] = str(folder.relative_to(ROOT))
    info = dict(name=name, settings={**exported['overrides'], 'label_smoothing': .05},
                folders=folders, best_filename='predictions.npz' if name == 'BASE' else 'best_predictions.npz',
                has_last_checkpoint=name != 'BASE', reference_config=exported['config'],
                tenfold_metrics=analysis['full'][name], round_two_plan_sha256=sha(ROUND2/'PLAN.json'),
                round_two_independent_audit_sha256=sha(ROUND2/'independent_audit.json'))
    return info, plan


def settings(anchor):
    specs = {'ANCHOR': dict(anchor)}
    aliases = {}
    for name, change in CHANGES.items():
        spec = {**anchor, **change}
        duplicate = next((old for old, value in specs.items() if value == spec), None)
        if duplicate is not None:
            aliases[name] = duplicate
        else:
            specs[name] = spec
    return specs, aliases


def folder_for(name, fold):
    require(ANCHOR_INFO is not None, 'Anchor has not been frozen')
    if name == 'ANCHOR':
        return ROOT/ANCHOR_INFO['folders'][str(fold)]
    return OUT/'runs'/name/f'fold_{fold:02d}'


def collect(name, fold):
    folder = folder_for(name, fold)
    record = read(folder/'complete.json')
    with (folder/'epochs.csv').open(encoding='utf-8') as stream:
        curve = list(csv.DictReader(stream))
    require([int(row['epoch']) for row in curve] == list(range(1, 1001)), 'Incomplete epoch history')
    acc = np.asarray([float(row['test_acc']) for row in curve])
    require(int(acc.argmax())+1 == record['best']['epoch'], 'First-maximum checkpoint rule changed')
    value = dict(name=name, fold=fold, best_epoch=record['best']['epoch'],
                 parameters=record['parameters'], late100_acc=float(acc[-100:].mean()))
    best_file = ANCHOR_INFO['best_filename'] if name == 'ANCHOR' else 'best_predictions.npz'
    for stage, filename in (('best', best_file), ('last', 'last_predictions.npz')):
        with np.load(folder/filename) as saved:
            value[stage] = common.metrics(saved['logits'], saved['labels'])
        require(abs(value[stage]['acc']-acc[record['best']['epoch']-1 if stage == 'best' else -1]) < 1e-12,
                f'{name}/{fold}/{stage}: prediction and curve disagree')
    return value


def aggregate(name, folds):
    rows = [collect(name, fold) for fold in folds]
    keys = ('acc', 'macro_f1', 'balanced_acc', 'pmci_auc', 'pmci_ap', 'pmci_recall', 'smci_recall')
    value = {stage: {key: float(np.mean([row[stage][key] for row in rows])) for key in keys}
             for stage in ('best', 'last')}
    value.update(folds=list(folds), late100_acc=float(np.mean([r['late100_acc'] for r in rows])),
                 acc_std=float(np.std([r['best']['acc'] for r in rows])))
    return value


def rank(names, folds):
    def key(name):
        value = aggregate(name, folds)
        return (value['best']['acc'], value['best']['macro_f1'],
                value['best']['pmci_recall'], value['late100_acc'])
    return sorted(names, key=key, reverse=True)


def write_config(name, spec):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(ROOT/CONFIG, encoding='utf-8')
    for key, value in spec.items():
        section = 'Modal' if key in ('Drop_rate', 'input_noise_std') else 'Optim'
        parser.set(section, key, str(value))
    with (OUT/'configs'/f'{name}.ini').open('w', encoding='utf-8') as stream:
        parser.write(stream)


def prepare(cfg, dd, data, graphs, paths, specs, aliases, prior_plan):
    require(cfg.Device.type == 'cuda' and cfg.epochs == 1000 and cfg.seed == 0, 'Wrong training protocol')
    require(dd['Class_Names'] == ['SMCI', 'PMCI'] and dd['Sample_Num'] == 535, 'Wrong dataset/task')
    protocol = prior_plan['baseline_protocol']
    for key, value in protocol['original_config'].items():
        require(getattr(cfg, key) == value, f'Original configuration changed:{key}')
    for path in paths:
        require(sha(path) == protocol['data_sha256'][Path(path).name], f'Data changed:{path}')
    with np.load(ROUND2/'data_splits_graphs.npz') as saved:
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(saved['test_masks'], torch.stack([m for _, m in data['Mask']]).cpu().numpy())
        for index, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{index}'], graph['edge'].numpy())
            np.testing.assert_array_equal(saved[f'neighbors_{index}'], graph['neighbors'].numpy())
    require(hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest() == protocol['candidate_hash'],
            'Original five graph candidates changed')
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        require(plan['anchor'] == ANCHOR_INFO and read(OUT/'all_settings.json') == specs, 'Frozen loss search changed')
        require(plan['aliases'] == aliases, 'Candidate aliases changed')
        verify_manifest(plan['code_sha256'])
        verify_manifest(plan['reference_sha256'])
        require(sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256'], 'Frozen split archive changed')
        return plan
    OUT.mkdir(exist_ok=False)
    (OUT/'configs').mkdir()
    (OUT/'runs').mkdir()
    sources = {**prior_plan['code_sha256'], Path(__file__).name: sha(__file__)}
    references = dict(prior_plan['reference_sha256'])
    for filename in ('audit_tad_binary_joint_tune.py', 'audit_tad_binary_fine_tune.py'):
        references[filename] = sha(ROOT/filename)
    for filename in ('PLAN.json', 'all_settings.json', 'analysis.json', 'COMPLETE.json',
                     'independent_audit.json', 'best_configuration.json', 'selection.json',
                     'screen_vs_remaining.json', 'summary.json', 'data_splits_graphs.npz'):
        path = ROUND2/filename
        references[str(path.relative_to(ROOT))] = sha(path)
    anchor_config = Path(ANCHOR_INFO['reference_config'])
    references[str(anchor_config.relative_to(ROOT))] = sha(anchor_config)
    for fold in ALL_FOLDS:
        folder = folder_for('ANCHOR', fold)
        filenames = ['best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json',
                     ANCHOR_INFO['best_filename'], 'last_predictions.npz']
        if ANCHOR_INFO['has_last_checkpoint']:
            filenames += ['last.pt', 'effective_settings.json']
        for filename in filenames:
            path = folder/filename
            references[str(path.relative_to(ROOT))] = sha(path)
    shutil.copy2(ROUND2/'data_splits_graphs.npz', OUT/'data_splits_graphs.npz')
    candidates = [name for name in specs if name != 'ANCHOR']
    plan = dict(created_at=timestamp(), anchor=ANCHOR_INFO, requested_changes=CHANGES,
        aliases=aliases, candidate_names=candidates, screen_folds=SCREEN, seed=0, epochs=1000,
        initial_screen_new_runs=4*len(candidates), maximum_new_runs=10*len(candidates),
        fixed='C content,716396 parameters; anchor lr,decay,dropout,input noise,orthogonality; original data,five graphs,tenfold masks,seed0,class weights,OVR heads,Adam,graph decay0,clip1,20-epoch hold/cosine1000/minlr.0001,argmax decisions.',
        selection='First maximum TEST ACC within each fold. Four-fold candidate ranking by ACC,macroF1,PMCI recall,late100 ACC,stable insertion order. Top2 and every primary-ACC tie at the second-place cutoff within1e-12 fill remaining six folds. Compare ANCHOR and finalists on complete tenfold results using the same ranking; one configuration for all folds.',
        loss_factory='Process-local explicit label_smoothing passed to the unchanged MSCQGNLoss for main and both auxiliary CE heads. No shared source files are modified.',
        reproduction_entrypoint=REPRODUCTION_ENTRYPOINT, config_alone_reproduces_loss=False,
        reproduction_note=REPRODUCTION_NOTE,
        scope='Single-factor loss hyperparameters only. Historical test-selected and adaptive test-fold protocol retained; not independent validation or global optimum. This search is optional, not automatically scheduled by round two.',
        baseline_protocol=protocol, code_sha256=sources, reference_sha256=references,
        split_archive_sha256=sha(OUT/'data_splits_graphs.npz'),
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    for filename in sources:
        target = OUT/'source_snapshot'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'all_settings.json', specs)
    return plan


def criterion_for(spec, dd, device):
    activate_loss(spec['label_smoothing'])
    return loss_factory(dd, device, rate=spec['Loss_rate'], aux_loss_weight=spec['aux_loss_weight'])


def loss_record(criterion):
    return dict(label_smoothing=float(criterion.CE_loss.label_smoothing),
        auxiliary_label_smoothing={name: float(module.label_smoothing) for name, module in criterion.aux_loss_dict.items()},
        class_weights=criterion.weight.detach().cpu().tolist(),
        auxiliary_class_weights={name: module.weight.detach().cpu().tolist() for name, module in criterion.aux_loss_dict.items()},
        aux_loss_weight=float(criterion.aux_loss_weight), orthogonality_rate=float(criterion.rate),
        aux_supervision=criterion.aux_supervision, reduction=criterion.CE_loss.reduction,
        formula='main weighted CE + aux_loss_weight * sum(two weighted OVR CEs) + orthogonality_rate * original full-cohort orthogonality')


def preflight(cfg, dd, data, graphs, specs):
    if (OUT/'preflight.json').exists():
        require(read(OUT/'preflight.json')['passed'] is True, 'Saved preflight failed')
        require((OUT/'anchor_replay.json').exists(), 'Saved anchor replay is missing')
        return
    with (folder_for('ANCHOR', 1)/'epochs.csv').open(encoding='utf-8') as stream:
        expected = list(csv.DictReader(stream))[:3]
    checks, initial, rng, initial_logits = {}, None, None, None
    for name, spec in specs.items():
        conf = common.configure(cfg, spec)
        SET_Random(0)
        model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device)
        require(sum(p.numel() for p in model.parameters()) == 716396, 'C parameter count changed')
        if initial is None:
            initial = {key: value.detach().clone() for key, value in model.named_parameters()}
            rng = torch.get_rng_state()
        else:
            require(torch.equal(rng, torch.get_rng_state()), 'Initialization RNG differs')
            for key, value in model.named_parameters():
                torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            logits = model(data['Feature'])[0]
        if initial_logits is None:
            initial_logits = logits.detach().clone()
        else:
            torch.testing.assert_close(logits, initial_logits, atol=2e-7, rtol=1e-6)
        criterion = criterion_for(spec, dd, cfg.Device)
        effective = loss_record(criterion)
        require(effective['class_weights'] == dd['Label_Weight'].detach().cpu().tolist(), 'Class weights changed')
        opt = common.retained_optimizer(model, conf)
        require([group['weight_decay'] for group in opt.param_groups] == [spec['weight_decay'], 0.], 'Optimizer groups changed')
        sched = CustomCosineAnnealingLR(opt, T_max=conf.T_max, eta_min=conf.Lr_Min)
        losses = []
        for index in range(3):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(data['Feature'])
            loss = criterion(logits, data['Label'], data['Mask'][0][0], embeddings, auxiliary)
            require(torch.isfinite(loss).item(), 'Nonfinite preflight loss')
            if name == 'ANCHOR':
                field = 'training_loss' if 'training_loss' in expected[index] else 'train_loss'
                require(abs(float(loss.detach())-float(expected[index][field])) < 2e-5,
                        'Anchor smoothing=.05 does not reproduce its historical training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip, error_if_nonfinite=True)
            opt.step()
            _, assessment = common.runner.assess(model, data, 1, criterion)
            if name == 'ANCHOR':
                require(abs(assessment['test_acc']-float(expected[index]['test_acc'])) < 1e-12,
                        'Anchor preflight test accuracy changed')
            sched.step()
            losses.append(float(loss.detach()))
        checks[name] = dict(passed=True, parameters=716396, first3_losses=losses, effective_loss=effective)
        del model, criterion, opt, sched, logits, embeddings, auxiliary, loss
    # The module attributes and a manual expression both verify smoothing is
    # effective, including zero: no truthy fallback can silently replace it.
    manual_checks = {}
    fit = data['Mask'][0][0]
    z, y = initial_logits[fit], data['Label'][fit]
    logp = z.log_softmax(-1)
    weights = dd['Label_Weight']
    for value in sorted({spec['label_smoothing'] for spec in specs.values()}):
        criterion = criterion_for({**specs['ANCHOR'], 'label_smoothing': value}, dd, cfg.Device)
        nll = -logp[torch.arange(len(y), device=y.device), y]*weights[y]
        smooth = -(logp*weights).sum(-1)/2
        manual = ((1-value)*nll + value*smooth).sum()/weights[y].sum()
        actual = criterion.CE_loss(z, y)
        torch.testing.assert_close(actual, manual, atol=2e-7, rtol=2e-6)
        manual_checks[str(value)] = dict(actual=float(actual), manual=float(manual), passed=True)
    del initial, rng, initial_logits, criterion, z, y, logp, manual, actual
    conf = common.configure(cfg, specs['ANCHOR'])
    model = RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    native_noise = model._modal_noise_std.detach().clone()
    replay = {}
    for fold in ALL_FOLDS:
        folder = folder_for('ANCHOR', fold)
        mask = data['Mask'][fold-1][1]
        replay[str(fold)] = {}
        for stage in ('best', 'last'):
            filename = ANCHOR_INFO['best_filename'] if stage == 'best' else 'last_predictions.npz'
            with np.load(folder/filename) as saved:
                logits = saved['logits'].copy()
                np.testing.assert_array_equal(saved['labels'], data['Label'][mask].cpu().numpy())
                if 'row_index' in saved:
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[mask.cpu().numpy()])
            available = stage == 'best' or ANCHOR_INFO['has_last_checkpoint']
            error = None
            if available:
                cp = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
                require(cp['fold'] == fold and cp['seed'] == 0, 'Anchor checkpoint identity changed')
                if ANCHOR_INFO['name'] == 'BASE':
                    require(cp['mode'] == 'content', 'Anchor model changed')
                else:
                    require(cp['arm'] == ANCHOR_INFO['name'], 'Anchor arm changed')
                model.load_state_dict(cp['state_dict'], strict=True)
                torch.testing.assert_close(model._modal_noise_std, native_noise, atol=0, rtol=0)
                with torch.inference_mode():
                    restored = model(data['Feature'])[0][mask].cpu().numpy()
                np.testing.assert_allclose(restored, logits, atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(restored.argmax(1), logits.argmax(1))
                error = float(np.abs(restored-logits).max())
                del cp, restored
            replay[str(fold)][stage] = dict(checkpoint_replayed=available, maximum_logit_error=error)
    require(aggregate('ANCHOR', ALL_FOLDS) == ANCHOR_INFO['tenfold_metrics'], 'Anchor tenfold metrics differ')
    dump(OUT/'anchor_replay.json', dict(passed=True, source_name=ANCHOR_INFO['name'], folds=replay,
        limitation=None if ANCHOR_INFO['has_last_checkpoint'] else 'Original BASE has no last.pt; last prediction labels/metrics verified without checkpoint replay.'))
    dump(OUT/'preflight.json', dict(passed=True, initialization_identical=True, checks=checks,
        smoothing_manual_formula_checks=manual_checks, anchor_smoothing_05_first3_reproduced=True))
    del model, logits
    torch.cuda.empty_cache()


def report(specs):
    summary = {}
    lines = ['# TAD 二分类：固定第二轮赢家的单因素损失参数搜索', '',
        f"ANCHOR来源：{ANCHOR_INFO['name']}；固定seed0、原十折、1000轮、首次最高测试ACC选轮。",
        '四折只用于探索筛选；每个候选仅改变标签平滑或辅助损失权重之一。',
        '复现必须使用 tad_binary_loss_tune.py 的显式损失工厂，或向 MSCQGNLoss 显式传入记录的标签平滑值，作用于主分类和两个辅助分类损失。旧 Config_ 和训练入口不读取新增标签平滑配置，仅使用导出 INI 不能复现该损失设置。', '',
        '| 配置 | 已完成折数 | 最佳ACC | 宏F1 | PMCI召回 | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|']
    for name in specs:
        folds = [fold for fold in ALL_FOLDS if (folder_for(name, fold)/'complete.json').exists()]
        if not folds:
            continue
        value = aggregate(name, folds)
        summary[name] = value
        lines.append(f"| {name} | {len(folds)}/10 | {100*value['best']['acc']:.4f}% | {100*value['best']['macro_f1']:.4f}% | {100*value['best']['pmci_recall']:.2f}% | {100*value['last']['acc']:.4f}% |")
    dump(OUT/'summary.json', dict(updated_at=timestamp(), candidates=summary))
    (OUT/'进度与结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_one(name, fold, specs, cfg, dd, data, graphs, phase):
    spec = specs[name]
    conf = common.configure(cfg, spec)
    folder = folder_for(name, fold)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, value in (('effective_settings.json', spec),
                            ('effective_loss.json', loss_record(criterion_for(spec, dd, cfg.Device)))):
        if (folder/filename).exists():
            require(read(folder/filename) == value, f'Effective training setting changed:{folder}/{filename}')
        else:
            dump(folder/filename, value)
    dump(OUT/'status.json', dict(state=phase, arm=name, fold=fold, settings=spec, updated_at=timestamp()))
    result = common.runner.run(name, fold, conf, dd, data, graphs)
    require(result['audit_passed'] and result['parameters'] == 716396, 'Training replay failed')
    require([g['weight_decay'] for g in read(folder/'optimizer_groups.json')] == [spec['weight_decay'], 0.],
            'Optimizer decay groups changed')
    with np.load(folder/'best_predictions.npz') as saved:
        mask = data['Mask'][fold-1][1]
        np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[mask.cpu().numpy()])
        np.testing.assert_array_equal(saved['labels'], data['Label'][mask].cpu().numpy())
    report(specs)
    torch.cuda.empty_cache()


def finish(specs, finalists, plan, dd):
    names = ['ANCHOR']+finalists
    retained = rank(names, ALL_FOLDS)[0]
    full = {name: aggregate(name, ALL_FOLDS) for name in names}
    pairs, records = {}, []
    for name in names:
        logits, labels, ids, old_logits = [], [], [], []
        for fold in ALL_FOLDS:
            filename = ANCHOR_INFO['best_filename'] if name == 'ANCHOR' else 'best_predictions.npz'
            with np.load(folder_for(name, fold)/filename) as saved, np.load(folder_for('ANCHOR', fold)/ANCHOR_INFO['best_filename']) as old:
                np.testing.assert_array_equal(saved['labels'], old['labels'])
                np.testing.assert_array_equal(saved['row_index'], old['row_index'])
                logits.extend(saved['logits']); labels.extend(saved['labels']); ids.extend(saved['row_index']); old_logits.extend(old['logits'])
            records.append(collect(name, fold))
        require(len(ids) == len(set(ids)) == dd['Sample_Num'], 'OOF rows are missing or repeated')
        np.testing.assert_array_equal(sorted(ids), sorted(dd['Index']))
        correct = np.asarray(logits).argmax(1) == labels
        old_correct = np.asarray(old_logits).argmax(1) == labels
        pairs[name] = dict(corrected=int((correct & ~old_correct).sum()), harmed=int((~correct & old_correct).sum()),
                           pooled=common.metrics(logits, labels))
        if name == retained:
            np.savez_compressed(OUT/'retained_oof_predictions.npz', labels=labels, logits=logits, row_index=ids)
    new_runs = list((OUT/'runs').glob('*/fold_*/complete.json'))
    require(len(new_runs) == plan['initial_screen_new_runs']+6*len(finalists) <= plan['maximum_new_runs'],
            'Prespecified run count differs')
    require(all(read(path)['audit_passed'] for path in new_runs), 'A new training audit failed')
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    dump(OUT/'analysis.json', dict(retained=retained, anchor_source=ANCHOR_INFO['name'], full=full,
        pairs=pairs, records=records, settings=specs[retained], new_runs=len(new_runs),
        verified_new_checkpoints=2*len(new_runs)))
    dump(OUT/'screen_vs_remaining.json', {name: dict(screen=aggregate(name, SCREEN),
         remaining_six=aggregate(name, REMAINING)) for name in names})
    dump(OUT/'best_configuration.json', dict(name=retained, anchor_source=ANCHOR_INFO['name'],
        config=str(OUT/'configs'/f'{retained}.ini'), overrides=specs[retained],
        model='RelationSheafModel(mode=content)', graph_weight_decay=0., k=10, seed=0, epochs=1000,
        parameters=716396, loss_factory='Explicit label_smoothing for both main and auxiliary CE',
        reproduction_entrypoint=REPRODUCTION_ENTRYPOINT, config_alone_reproduces_loss=False,
        reproduction_note=REPRODUCTION_NOTE,
        selection='First maximum TEST ACC per fold; one unified configuration; bounded exploratory search, not global optimum.'))
    report(specs)
    with (OUT/'进度与结果.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n完整十折保留：{retained}。新增{len(new_runs)}次训练、{2*len(new_runs)}个新检查点训练内复核通过；独立审计另行运行。\n')
        stream.write('所有结果仍是原测试选轮与自适应测试折搜索下的探索性结果。每行数据一次折外预测不等同于独立临床患者ID确认。\n')
    dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), all_audits_passed=True,
                                 new_runs=len(new_runs), retained=retained))
    dump(OUT/'status.json', dict(state='complete', retained=retained, new_runs=len(new_runs),
                               finalists=finalists, updated_at=timestamp()))


def main():
    global ANCHOR_INFO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    ANCHOR_INFO, prior_plan = prerequisite()
    require(not (OUT/'RUNNING.lock').exists(), 'The loss search already has a training lock')
    specs, aliases = settings(ANCHOR_INFO['settings'])
    cfg, dd, data, graphs, paths = setup(0, CONFIG, 10)
    plan = prepare(cfg, dd, data, graphs, paths, specs, aliases, prior_plan)
    for name, spec in specs.items():
        write_config(name, spec)
    common.runner.OUT = OUT/'runs'
    common.runner.relation_optimizer = common.retained_optimizer
    common.runner.MSCQGNLoss = loss_factory
    preflight(cfg, dd, data, graphs, specs)
    if (OUT/'COMPLETE.json').exists():
        return
    report(specs)
    if args.prepare_only:
        print(json.dumps(dict(preflight=True, anchor=ANCHOR_INFO['name'],
            maximum_new_runs=plan['maximum_new_runs'], output=str(OUT))), flush=True)
        return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        candidates = plan['candidate_names']
        require(len(candidates) >= 2, 'Expected at least two nonduplicate loss settings')
        for name in candidates:
            for fold in SCREEN:
                run_one(name, fold, specs, cfg, dd, data, graphs, 'loss_four_fold_screen')
        order = rank(candidates, SCREEN)
        cutoff = aggregate(order[1], SCREEN)['best']['acc']
        finalists = [name for name in order if aggregate(name, SCREEN)['best']['acc'] >= cutoff-1e-12]
        dump(OUT/'selection.json', dict(candidate_order=order, reference_order=rank(list(specs), SCREEN),
            cutoff_acc=cutoff, finalists=finalists, tie_rule='all four-fold primary-ACC ties at second place advance'))
        for name in finalists:
            for fold in REMAINING:
                run_one(name, fold, specs, cfg, dd, data, graphs, 'loss_ten_fold')
        finish(specs, finalists, plan, dd)
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
