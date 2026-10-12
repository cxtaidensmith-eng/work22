"""Complete the sole omitted pooled-ACC screening tie, N005, on ABIDE-5.

This is a separate, explicitly documented extension of the finished main search.
It reuses four N005 folds and trains exactly six missing folds without rewriting
the original plan, selection, results or source files. Import does not use CUDA.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import abide5_fine_tune as lib
from alternating_cooperative_experiment import read, verify_manifest
from relation_experiment import ROOT, dump, setup, sha, timestamp

OUT = ROOT/'Result/abide5_tie_completion_20260928'
MAIN = ROOT/'Result/abide5_fine_tune_20260928'
BASE_REF = ROOT/'Result/c_abide5_seed0_20260924'
SCREEN = (1, 3, 6, 8)
MISSING = (2, 4, 5, 7, 9, 10)
FOLDS = tuple(range(1, 11))
N005 = {**lib.BASE, 'input_noise_std': .05}
CONFIG = 'Config/c_abide5_transfer.ini'


def folder_for(name, fold):
    if name == 'BASE':
        root = BASE_REF/'content'
    elif name in ('N015', 'A050') or (name == 'N005' and fold in SCREEN):
        root = MAIN/'runs'/name
    elif name == 'N005' and fold in MISSING:
        root = OUT/'runs'/name
    else:
        raise ValueError((name, fold))
    return root/f'fold_{fold:02d}'


def bind_library():
    # These are interpreter-local assignments, not edits to frozen source files.
    lib.OUT = OUT
    lib.folder_for = folder_for
    lib.runner.OUT = OUT/'runs'
    lib.runner.RelationSheafModel = lib.RelationSheafModel
    lib.runner.relation_optimizer = lib.retained_optimizer


def screening_correct(name):
    correct, samples = 0, 0
    for fold in SCREEN:
        if name == 'BASE':
            folder, filename = BASE_REF/'content'/f'fold_{fold:02d}', 'predictions.npz'
        else:
            folder, filename = MAIN/'runs'/name/f'fold_{fold:02d}', 'best_predictions.npz'
        with np.load(folder/filename) as saved:
            correct += int((saved['logits'].argmax(1) == saved['labels']).sum())
            samples += len(saved['labels'])
    assert samples == 346
    return correct


def prepare(cfg, dd, data, graphs, paths):
    assert (MAIN/'COMPLETE.json').exists() and not (MAIN/'RUNNING.lock').exists()
    main_complete = read(MAIN/'COMPLETE.json')
    assert main_complete['all_audits_passed'] and main_complete['new_runs'] == 72
    main_audit = read(MAIN/'independent_audit.json')
    assert main_audit['passed'] and main_audit['new_runs'] == 72
    main_plan = read(MAIN/'PLAN.json')
    verify_manifest(main_plan['code_sha256'])
    verify_manifest(main_plan['reference_sha256'])
    for filename, expected in main_audit['artifact_sha256'].items():
        assert sha(MAIN/filename) == expected, filename
    assert sha(ROOT/'audit_abide5_fine_tune.py') == main_audit['auditor_sha256']
    settings = read(MAIN/'all_settings.json')
    assert settings['N005'] == N005 and settings['BASE'] == lib.BASE
    selection = read(MAIN/'selection.json')
    assert selection['finalists'] == ['N015', 'A050']
    counts = {name: screening_correct(name) for name in settings}
    maximum = max(counts.values())
    eligible = [name for name in settings if name != 'BASE'
                and name not in selection['finalists'] and counts[name] >= maximum]
    assert maximum == 315 and eligible == ['N005']
    # Keep the candidate insertion order from the already frozen original search.
    specs = {name: spec for name, spec in settings.items() if name in ('BASE', 'N005', 'N015', 'A050')}
    assert list(specs) == ['BASE', 'N005', 'N015', 'A050']
    assert cfg.Device.type == 'cuda' and cfg.epochs == 1000 and cfg.seed == 0
    assert dd['Sample_Num'] == 864 and dd['Class_Names'] == ['ADS', 'CN']
    assert dd['Modal_Name'] == ['PHENO', 'ANAT', 'FUNC', 'MRI', 'fMRI']
    assert list(map(len, dd['Modal_Index'])) == [48, 6, 10, 150, 256]
    assert torch.bincount(data['Label']).cpu().tolist() == [397, 467]
    for key, value in main_plan['baseline_protocol']['original_config'].items():
        assert getattr(cfg, key) == value, key
    for path in paths:
        assert sha(path) == main_plan['baseline_protocol']['data_sha256'][Path(path).name]
    lib.check_archive(MAIN/'data_splits_graphs.npz', dd, data, graphs)
    assert main_plan['baseline_protocol']['candidate_hash'] == hashlib.sha256(
        b''.join(graph['edge'].numpy().tobytes() for graph in graphs)).hexdigest()
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        assert plan['specs'] == specs and plan['screen_correct_counts'] == counts
        assert read(OUT/'all_settings.json') == specs
        assert plan['new_folds'] == list(MISSING) and plan['reused_n005_folds'] == list(SCREEN)
        verify_manifest(plan['code_sha256'])
        verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
        lib.check_archive(OUT/'data_splits_graphs.npz', dd, data, graphs)
        return plan, specs
    (OUT/'configs').mkdir(exist_ok=True)
    (OUT/'runs').mkdir(exist_ok=True)
    sources = dict(main_plan['code_sha256'])
    sources[Path(__file__).name] = sha(__file__)
    references = dict(main_plan['reference_sha256'])
    for filename in ('PLAN.json', 'COMPLETE.json', 'independent_audit.json', 'analysis.json',
                     'all_settings.json', 'selection.json', 'refinement.json', 'summary.json',
                     'preflight.json', 'config_sha256.json', 'screen_vs_remaining.json',
                     'best_configuration.json', 'retained_oof_predictions.npz', 'data_splits_graphs.npz'):
        path = MAIN/filename
        references[str(path.relative_to(ROOT))] = sha(path)
    # All screening predictions are pinned: the choice of N005 is reproducible,
    # including the fact that there is no other omitted equal-correct candidate.
    for name in settings:
        if name == 'BASE':
            continue
        for fold in SCREEN:
            for filename in ('best_predictions.npz', 'complete.json'):
                path = MAIN/'runs'/name/f'fold_{fold:02d}'/filename
                references[str(path.relative_to(ROOT))] = sha(path)
    reused_folds = []
    for name in specs:
        folds = SCREEN if name == 'N005' else FOLDS
        for fold in folds:
            folder = folder_for(name, fold)
            filenames = ('best.pt', 'predictions.npz', 'last_predictions.npz',
                         'epochs.csv', 'complete.json', 'optimizer_groups.json') if name == 'BASE' else (
                         'best.pt', 'last.pt', 'best_predictions.npz', 'last_predictions.npz',
                         'epochs.csv', 'complete.json', 'optimizer_groups.json',
                         'effective_settings.json', 'fine_audit.json')
            for filename in filenames:
                path = folder/filename
                references[str(path.relative_to(ROOT))] = sha(path)
            reused_folds.append(dict(name=name, fold=fold, folder=str(folder.relative_to(ROOT))))
        path = MAIN/'configs'/f'{name}.ini'
        references[str(path.relative_to(ROOT))] = sha(path)
    shutil.copy2(MAIN/'data_splits_graphs.npz', OUT/'data_splits_graphs.npz')
    plan = dict(created_at=timestamp(), extension='Complete omitted pooled-correct screening tie N005 only',
        reason='N005, N015, A050 and BASE each classified315/346 screening rows correctly. '
               'N005 was below the original equal-fold ACC cutoff by only0.003341 percentage points '
               'because test folds contain86/87 rows. N005 is the sole omitted candidate tied in '
               'total correct predictions. Complete its six missing folds without changing original results.',
        selection_rule='From all frozen original settings, take all nonBASE, nonfinalist candidates '
                       'whose pooled screening correct count reaches the maximum across original settings; '
                       'the realized eligible list must be exactly[N005]. No additional parameters or candidates.',
        screen_correct_counts=counts, eligible=eligible, specs=specs,
        source_experiment=str(MAIN.relative_to(ROOT)), base_reference=str(BASE_REF.relative_to(ROOT)),
        screen_folds=list(SCREEN), reused_n005_folds=list(SCREEN), new_folds=list(MISSING),
        new_runs=6, reused_n005_runs=4, original_search_new_runs=72,
        total_search_plus_extension_new_runs=78, compare_order=list(specs),
        seed=0, epochs=1000, parameters=768348, samples=864, counts=[397, 467],
        final_rank='Mean first-best TEST ACC, macroF1, balancedACC, late100 ACC, each rounded12, '
                   'then original candidate insertion order BASE,N005,N015,A050. One setting per tenfold.',
        fixed='Original C content architecture, 768348 parameters, processed ABIDE-5 data, seed0 '
              'tenfold masks, five top10 graph supports, class weights, smoothing.05, graph decay0 '
              'for N005, gate priors, hold20/cosine1000/minlr.0001, Adam and clipping1. '
              'N005 has input_noise_std.05, actual noise.15 in each modality; other settings equal BASE.',
        scope='Separate disclosed post-search extension. All test-selected estimates remain exploratory; '
              'no independent validation, seed replication, threshold adjustment or global-optimum claim. '
              'Stop after the prescribed six folds even if BASE remains best.',
        original_plan_sha256=sha(MAIN/'PLAN.json'), original_selection=selection,
        baseline_protocol=main_plan['baseline_protocol'], reused_folds=reused_folds,
        code_sha256=sources, reference_sha256=references,
        split_archive_sha256=sha(OUT/'data_splits_graphs.npz'))
    for filename in sources:
        target = OUT/'source_snapshot'/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    dump(OUT/'PLAN.json', plan)
    dump(OUT/'all_settings.json', specs)
    return plan, specs


def preflight(cfg, dd, data, graphs, specs):
    lib.preflight(cfg, dd, data, graphs, specs)
    result = read(OUT/'preflight.json')
    assert result['passed'] and set(result['checks']) == set(specs)
    original = read(MAIN/'preflight.json')['checks']['N005']
    current = result['checks']['N005']
    for field in ('spec', 'parameters', 'init_parameter_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256',
                  'optimizer_groups', 'effective_noise_sd'):
        assert current[field] == original[field], field
    with (folder_for('N005', 1)/'epochs.csv').open(encoding='utf-8') as stream:
        curve = list(csv.DictReader(stream))[:3]
    trace = []
    for new, old, actual in zip(current['first3'], original['first3'], curve):
        # Compare to the real training trajectory with the same established limit.
        difference = abs(new['loss']-float(actual['training_loss']))
        assert difference <= 2e-5
        assert abs(new['test_acc']-float(actual['test_acc'])) < 1e-12
        assert new['epoch'] == old['epoch'] == int(actual['epoch'])
        assert abs(new['lr']-float(actual['lr'])) < 1e-12
        trace.append(dict(epoch=new['epoch'], loss=new['loss'], training_loss=float(actual['training_loss']),
                          loss_absolute_difference=difference,
                          old_preflight_loss_absolute_difference=abs(new['loss']-old['loss']),
                          test_acc=new['test_acc']))
    # Reused N005 fourfold weights are replayed without writing to original folders.
    conf = lib.configure(cfg, N005)
    model = lib.RelationSheafModel(dd, conf, graphs, 'content', training_seed=0).to(cfg.Device).eval()
    checks = []
    for fold in SCREEN:
        folder = folder_for('N005', fold)
        assert read(folder/'effective_settings.json') == N005
        record = lib.collect('N005', fold)
        test = data['Mask'][fold-1][1]
        for stage in ('best', 'last'):
            checkpoint = torch.load(folder/f'{stage}.pt', map_location=cfg.Device, weights_only=False)
            assert (checkpoint['arm'], checkpoint['fold'], checkpoint['seed']) == ('N005', fold, 0)
            if stage == 'best':
                assert checkpoint['best']['epoch'] == record['best_epoch']
            else:
                assert checkpoint['epoch'] == 1000
            model.load_state_dict(checkpoint['state_dict'])
            torch.testing.assert_close(model._modal_noise_std.cpu(), torch.full((5,), .3), atol=0, rtol=0)
            with torch.no_grad():
                logits = model(data['Feature'])[0][test].cpu().numpy()
            with np.load(folder/f'{stage}_predictions.npz') as saved:
                np.testing.assert_allclose(logits, saved['logits'], atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(logits.argmax(1), saved['logits'].argmax(1))
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                if stage == 'best':
                    np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
            checks.append(dict(fold=fold, stage=stage, passed=True, checkpoint_sha256=sha(folder/f'{stage}.pt')))
    dump(OUT/'reuse_preflight.json', dict(passed=True, n005_first_three_vs_original_training=trace,
        reused_n005_checkpoints=checks, verified_reused_n005_checkpoints=8,
        initialization_rng_optimizer_noise_match_original=True))
    del model
    torch.cuda.empty_cache()


def finish(specs, plan, dd):
    names = list(specs)
    order = lib.rank(names, FOLDS)
    retained = order[0]
    full = {name: lib.aggregate(name, FOLDS) for name in names}
    pairs, rows, stages = {}, [], {}
    (OUT/'oof').mkdir(exist_ok=True)
    for name in names:
        ys, logits, indices, folds = [], [], [], []
        corrected = harmed = wins = ties = losses = 0
        for fold in FOLDS:
            filename = 'predictions.npz' if name == 'BASE' else 'best_predictions.npz'
            with np.load(folder_for(name, fold)/filename) as saved, np.load(
                    folder_for('BASE', fold)/'predictions.npz') as base:
                np.testing.assert_array_equal(saved['labels'], base['labels'])
                np.testing.assert_array_equal(saved['row_index'], base['row_index'])
                correct = saved['logits'].argmax(1) == saved['labels']
                old = base['logits'].argmax(1) == base['labels']
                corrected += int((correct & ~old).sum())
                harmed += int((~correct & old).sum())
                delta = int(correct.sum())-int(old.sum())
                wins += int(delta > 0)
                ties += int(delta == 0)
                losses += int(delta < 0)
                ys.extend(saved['labels'])
                logits.extend(saved['logits'])
                indices.extend(saved['row_index'])
                folds.extend([fold]*len(saved['labels']))
            rows.append(lib.collect(name, fold))
        assert len(indices) == len(set(indices)) == dd['Sample_Num'] == 864
        np.testing.assert_array_equal(sorted(indices), sorted(dd['Index']))
        pairs[name] = dict(corrected=corrected, harmed=harmed, wins=wins, ties=ties, losses=losses,
                           pooled=lib.metrics(logits, ys))
        np.savez_compressed(OUT/'oof'/f'{name}.npz', labels=ys, logits=logits, row_index=indices, fold=folds)
        if name == retained:
            shutil.copy2(OUT/'oof'/f'{name}.npz', OUT/'retained_oof_predictions.npz')
        stages[name] = dict(screen=lib.aggregate(name, SCREEN), remaining=lib.aggregate(name, MISSING),
            screen_delta_acc=lib.aggregate(name, SCREEN)['best']['acc']-lib.aggregate('BASE', SCREEN)['best']['acc'],
            remaining_delta_acc=lib.aggregate(name, MISSING)['best']['acc']-lib.aggregate('BASE', MISSING)['best']['acc'])
    original = read(MAIN/'analysis.json')
    for name in ('BASE', 'N015', 'A050'):
        assert full[name] == original['full'][name]
        assert pairs[name] == original['pairs'][name]
    new_paths = list((OUT/'runs').glob('*/fold_*/complete.json'))
    assert len(new_paths) == 6
    assert {(path.parent.parent.name, int(path.parent.name.split('_')[1])) for path in new_paths} == {
        ('N005', fold) for fold in MISSING}
    for path in new_paths:
        assert read(path)['audit_passed'] and read(path.parent/'fine_audit.json')['passed']
    for name, spec in specs.items():
        lib.write_config(name, spec)
    verify_manifest(plan['code_sha256'])
    verify_manifest(plan['reference_sha256'])
    assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
    analysis = dict(retained=retained, full_order=order, full=full, pairs=pairs, records=rows,
        settings=specs[retained], screen_vs_remaining=stages, new_runs=6, verified_new_checkpoints=12,
        reused_n005_runs=4, completed_n005_tenfold_runs=10,
        original_search_new_runs=72, total_search_plus_extension_new_runs=78)
    dump(OUT/'analysis.json', analysis)
    dump(OUT/'screen_vs_remaining.json', stages)
    dump(OUT/'best_configuration.json', dict(name=retained, config=str(OUT/'configs'/f'{retained}.ini'),
        driver='abide5_tie_completion.py', base_config=CONFIG, overrides=specs[retained],
        model='RelationSheafModel(mode=content)', graph_weight_decay=specs[retained]['graph_weight_decay'],
        k=10, seed=0, epochs=1000, parameters=768348,
        effective_noise_sd=[3*specs[retained]['input_noise_std']]*5,
        reproduction='Use lib.configure(cfg,overrides) and lib.retained_optimizer(model,cfg); '
                     'old Config_ ignores graph_weight_decay in exported INI.',
        selection='First maximum TEST ACC, one configuration per tenfold, exploratory completed comparison only.'))
    lib.report(specs)
    lines = ['# ABIDE-5：N005筛选并列候选补齐十折', '',
        '主搜索计划与结果保持原样。本扩展只补N005的第2、4、5、7、9、10折，复用原第1、3、6、8折。',
        '四折合计正确数的并列领先候选中，N005是原等权折均值规则未晋级的唯一候选。补齐属于单独披露的后续探索，不是修改原晋级结果。', '',
        '| 配置 | 十折最佳ACC | 宏F1 | 平衡ACC | ADS召回 | CN召回 | ADS概率AUC | 第1000轮ACC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in order:
        value = full[name]
        metrics = [value['best'][key] for key in ('acc', 'macro_f1', 'balanced_acc', 'ads_recall', 'cn_recall', 'ads_auc')]
        metrics.append(value['last']['acc'])
        lines.append('| '+name+' | '+' | '.join(f'{100*x:.4f}%' for x in metrics)+' |')
    lines += ['', f'最终保留：{retained}。新增6次训练、12个新检查点复核；N005十折包含4次复用和6次新增，不能计成10次新增。',
        '主搜索72次加本扩展6次，共78次新增完整训练。',
        '固定C网络768348参数、seed0、原十折、每折1000轮、首次最高测试ACC选轮；数据和图支持不变。',
        'N005仅把input_noise_std从0.1改为0.05，五模态真实噪声由0.3改为0.15。',
        '所有完整配置以864行对齐，合并指标与十折均值分开保存。', '',
        '| 折 | N005来源 | 最佳轮次 | BASE最佳ACC | N005最佳ACC | N005第1000轮ACC |',
        '|---|---|---:|---:|---:|---:|']
    for fold in FOLDS:
        row = lib.collect('N005', fold)
        old = lib.collect('BASE', fold)
        lines.append(f"| {fold} | {'复用' if fold in SCREEN else '新增'} | {row['best_epoch']} | "
                     f"{100*old['best']['acc']:.4f}% | {100*row['best']['acc']:.4f}% | {100*row['last']['acc']:.4f}% |")
    pair = pairs['N005']
    lines += ['', f'N005相对BASE纠正{pair["corrected"]}例、新增{pair["harmed"]}例错误；'
              f'{pair["wins"]}折提高、{pair["ties"]}折持平、{pair["losses"]}折下降。',
        f'N005筛选四折变化{100*stages["N005"]["screen_delta_acc"]:+.4f}个百分点，剩余六折变化'
        f'{100*stages["N005"]["remaining_delta_acc"]:+.4f}个百分点。',
        '本次补齐后按冻结计划结束，不再因结果不涨而追加新网格。',
        '所有成绩仍来自测试选轮与重复使用测试折的探索；未验证多种子或独立泛化，不能宣称全局最优。']
    (OUT/'补齐结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    dump(OUT/'COMPLETE.json', dict(completed_at=timestamp(), all_audits_passed=True,
        retained=retained, new_runs=6, verified_new_checkpoints=12, reused_n005_runs=4,
        completed_n005_tenfold_runs=10, original_search_new_runs=72, total_search_plus_extension_new_runs=78))
    dump(OUT/'status.json', dict(state='complete', retained=retained, new_runs=6, updated_at=timestamp()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    # Do not initialize CUDA while the original training/audit is incomplete.
    assert (MAIN/'COMPLETE.json').is_file() and not (MAIN/'RUNNING.lock').exists()
    assert (MAIN/'independent_audit.json').is_file()
    assert read(MAIN/'independent_audit.json')['passed']
    cfg, dd, data, graphs, paths = setup(0, CONFIG, 10)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        plan, specs = prepare(cfg, dd, data, graphs, paths)
        if (OUT/'COMPLETE.json').exists():
            return
        bind_library()
        for name, spec in specs.items():
            lib.write_config(name, spec)
        dump(OUT/'status.json', dict(state='preflight', updated_at=timestamp()))
        preflight(cfg, dd, data, graphs, specs)
        if args.prepare_only:
            lib.report(specs)
            dump(OUT/'status.json', dict(state='prepared', updated_at=timestamp()))
            print(json.dumps(dict(preflight=True, new_runs=6, reused_n005_runs=4, output=str(OUT))), flush=True)
            return
        for fold in MISSING:
            lib.run_one('N005', fold, specs, cfg, dd, data, graphs, 'complete_n005')
        dump(OUT/'status.json', dict(state='finalizing', updated_at=timestamp()))
        finish(specs, plan, dd)
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
