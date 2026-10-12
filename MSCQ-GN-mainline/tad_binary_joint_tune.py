"""Second TAD binary C search: four-fold joint tuning with tied-ACC promotion.

Only training hyperparameters change. The historical seed0/test-selected
tenfold protocol remains explicit; this is exploratory, not held-out evidence.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import tad_binary_fine_tune as common
from relation_experiment import ROOT, setup, dump, sha, timestamp
from alternating_cooperative_experiment import read, verify_manifest

OUT = ROOT/'Result/tad_binary_joint_tune_20260928'
PREV = ROOT/'Result/tad_binary_fine_tune_20260928'
REF = ROOT/'Result/c_tadpole_binary_seed0_20260924'
CONFIG = 'Config/c_tadpole_binary_transfer.ini'
SCREEN = (1, 3, 6, 8)
BASE = dict(common.BASE)
REFERENCE_NAMES = ('BASE', 'D050', 'N015')
CHANGES = {
    'O00001': dict(Loss_rate=.00001),
    'W002': dict(weight_decay=.002),
    'J_DN': dict(Drop_rate=.50, input_noise_std=.15),
    'J_MID': dict(Drop_rate=.57, input_noise_std=.125),
    'J_OW': dict(Loss_rate=.00001, weight_decay=.002),
    'J_LDN': dict(lr=.003, Drop_rate=.50, input_noise_std=.15),
    'J_DW': dict(Drop_rate=.70, weight_decay=.002),
}


def folder_for(name, fold):
    if name == 'BASE':
        return REF/'content'/f'fold_{fold:02d}'
    old = PREV/'runs'/name/f'fold_{fold:02d}'
    if (old/'complete.json').exists():
        return old
    return OUT/'runs'/name/f'fold_{fold:02d}'


def bind_helpers():
    # Reuse the audited loop, metric and artifact code with an explicit resolver.
    # Existing runs stay in their immutable first-round locations.
    common.OUT = OUT
    common.folder_for = folder_for
    common.runner.OUT = OUT/'runs'
    common.runner.relation_optimizer = common.retained_optimizer


def settings():
    previous = read(PREV/'all_settings.json')
    return {name: previous[name] for name in REFERENCE_NAMES} | {
        name: {**BASE, **change} for name, change in CHANGES.items()}


def prepare(cfg, dd, data, graphs, paths, specs):
    previous = read(PREV/'PLAN.json')
    baseline = read(REF/'PLAN.json')
    verify_manifest(previous['code_sha256']); verify_manifest(previous['reference_sha256'])
    assert read(PREV/'independent_audit.json')['passed']
    assert read(PREV/'COMPLETE.json')['all_audits_passed']
    assert not (PREV/'RUNNING.lock').exists()
    assert cfg.Device.type == 'cuda' and cfg.epochs == 1000 and cfg.seed == 0
    assert dd['Class_Names'] == ['SMCI', 'PMCI'] and dd['Sample_Num'] == 535
    for path in paths:
        assert sha(path) == baseline['protocol']['data_sha256'][Path(path).name]
    with np.load(PREV/'data_splits_graphs.npz') as p:
        np.testing.assert_array_equal(p['source_row'], dd['Index'])
        np.testing.assert_array_equal(p['labels'], data['Label'].cpu().numpy())
        np.testing.assert_array_equal(p['test_masks'], torch.stack([b for _, b in data['Mask']]).cpu().numpy())
        for index, graph in enumerate(graphs):
            np.testing.assert_array_equal(p[f'edge_{index}'], graph['edge'].numpy())
            np.testing.assert_array_equal(p[f'neighbors_{index}'], graph['neighbors'].numpy())
    assert hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest() == baseline['protocol']['candidate_hash']
    if (OUT/'PLAN.json').exists():
        plan = read(OUT/'PLAN.json')
        verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_sha256'])
        assert sha(OUT/'data_splits_graphs.npz') == plan['split_archive_sha256']
        assert read(OUT/'all_settings.json') == specs
        return plan
    OUT.mkdir(exist_ok=False); (OUT/'configs').mkdir(); (OUT/'runs').mkdir()
    sources = dict(previous['code_sha256'])
    sources[Path(__file__).name] = sha(__file__)
    references = dict(previous['reference_sha256'])
    reused = {}
    for name in specs:
        for fold in range(1, 11):
            folder = folder_for(name, fold)
            if not (folder/'complete.json').exists(): continue
            reused.setdefault(name, []).append(fold)
            filenames = ['best.pt', 'epochs.csv', 'complete.json', 'optimizer_groups.json',
                         'predictions.npz' if name == 'BASE' else 'best_predictions.npz', 'last_predictions.npz']
            if name != 'BASE': filenames += ['last.pt', 'effective_settings.json']
            for filename in filenames:
                p = folder/filename; references[str(p.relative_to(ROOT))] = sha(p)
    for filename in ('PLAN.json', 'all_settings.json', 'COMPLETE.json', 'independent_audit.json'):
        p = PREV/filename; references[str(p.relative_to(ROOT))] = sha(p)
    shutil.copy2(PREV/'data_splits_graphs.npz', OUT/'data_splits_graphs.npz')
    plan = dict(created_at=timestamp(), base=BASE, changes=CHANGES, reference_names=REFERENCE_NAMES,
        candidate_names=list(CHANGES), screen_folds=SCREEN, seed=0, epochs=1000,
        maximum_new_runs=66, initial_screen_new_runs=24, reused_folds=reused,
        fixed='C content network,716396 parameters,H96/depth5/heads4/own64; original processed data,five modalities,k10 graphs,train/test splits,class weights,label smoothing.05,Adam,cosine1000/minlr.0001,graph decay0,clip1,gate priors and all unlisted configuration.',
        selection='Rank by mean first-best-test ACC, macroF1, PMCI recall, late100 mean ACC, stable insertion order. Evaluate all seven candidates on folds1,3,6,8, reusing O00001/W002 folds1,6. Promote top2 candidates and every candidate tied in primary four-fold ACC with or above the second-ranked cutoff (tolerance1e-12); all promoted candidates fill remaining6 folds. Up to7 candidates/66 new runs. Full-tenfold final ranking includes BASE,D050,N015 and every finalist. One configuration is used across all folds. Four-fold losers are not asserted ineffective.',
        scope='Joint hyperparameter search, original test-selected protocol; adaptive reused test folds, no independent validation or global optimum claim. No architecture, graph support, threshold, class-weight, multiple-seed or per-fold configuration changes.',
        baseline_protocol=baseline['protocol'], code_sha256=sources, reference_sha256=references,
        split_archive_sha256=sha(OUT/'data_splits_graphs.npz'),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    for filename in sources:
        target = OUT/'source_snapshot'/filename; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    dump(OUT/'PLAN.json', plan); dump(OUT/'all_settings.json', specs)
    return plan


def run_one(name, fold, specs, cfg, dd, data, graphs, phase):
    folder = folder_for(name, fold)
    if (folder/'complete.json').exists() and not folder.is_relative_to(OUT):
        # Already audited and hash-pinned first-round training is reused.
        return
    common.run_one(name, fold, specs, cfg, dd, data, graphs, phase)


def finalize(specs, finalists, plan, dd):
    # common.finish prepends BASE; include the other two completed references.
    compared = ['D050', 'N015']+finalists
    common.finish(specs, compared, plan, dd)
    analysis = read(OUT/'analysis.json')
    new_count = len(list((OUT/'runs').glob('*/fold_*/complete.json')))
    assert new_count == 24+6*len(finalists)
    detail = {}
    for name in ['BASE']+compared:
        screen = common.aggregate(name, SCREEN)
        remaining = common.aggregate(name, tuple(f for f in range(1, 11) if f not in SCREEN))
        detail[name] = dict(screen=screen, remaining_six=remaining)
    dump(OUT/'screen_vs_remaining.json', detail)
    with (OUT/'进度与结果.md').open('a', encoding='utf-8') as stream:
        stream.write('\n本轮筛选使用第1、3、6、8折；完整结果另外包含其余六折。原始候选全部披露，同ACC晋级边界的候选也补齐十折。\n')
    dump(OUT/'status.json', dict(state='complete', retained=analysis['retained'], new_runs=new_count,
         finalists=finalists, updated_at=timestamp()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    cfg, dd, data, graphs, paths = setup(0, CONFIG, 10)
    specs = settings(); plan = prepare(cfg, dd, data, graphs, paths, specs)
    bind_helpers()
    for name, spec in specs.items(): common.write_config(name, spec)
    common.preflight(cfg, dd, data, graphs, specs)
    common.report(specs)
    if args.prepare_only:
        print(json.dumps(dict(preflight=True, maximum_new_runs=66, output=str(OUT))), flush=True); return
    if (OUT/'COMPLETE.json').exists(): return
    lock = OUT/'RUNNING.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try:
        for name in CHANGES:
            for fold in SCREEN: run_one(name, fold, specs, cfg, dd, data, graphs, 'four_fold_screen')
        order = common.rank(list(CHANGES), SCREEN)
        cutoff = common.aggregate(order[1], SCREEN)['best']['acc']
        finalists = [name for name in order if common.aggregate(name, SCREEN)['best']['acc'] >= cutoff-1e-12]
        dump(OUT/'selection.json', dict(candidate_order=order, reference_order=common.rank(list(specs), SCREEN),
             cutoff_acc=cutoff, finalists=finalists, tie_rule='all four-fold ACC ties at second place advance'))
        for name in finalists:
            for fold in range(1, 11):
                if fold not in SCREEN: run_one(name, fold, specs, cfg, dd, data, graphs, 'ten_fold')
        finalize(specs, finalists, plan, dd)
    except BaseException as error:
        dump(OUT/'status.json', dict(state='failed', error=repr(error), updated_at=timestamp())); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
