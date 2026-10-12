"""One preregistered cross-relation null control for retained TAD C.

Only XREL_NULL is trained. C and D1 are read-only historical references.
The null randomizes cross-relation topology AND placement of raw priors; this
is not a pure unweighted-topology effect or a validation-selected benchmark.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch

from relation_experiment import ROOT, dump, sha, timestamp
from alternating_cooperative_experiment import (
    base_setup, read, state_hash, rng_state, assert_rng_equal, metrics,
    verify_manifest, verify_reference, save_predictions)
from analyze_joint_evidence_routing import _record, _aggregate, _pair
from Model.relation_sheaf import RelationSheafModel
from Model.c_cross_relation_null import CrossRelationNullModel, build_null_graphs
from Loss import MSCQGNLoss
from Utils import SET_Random, CustomCosineAnnealingLR
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


OUT = ROOT / 'Result/c_cross_relation_null_seed0_20260928'
REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0/content'
D1_REFERENCE = ROOT / 'Result/c_mechanism_controls_seed0_20260924/seed_0/D1'
MODE = 'XREL_NULL'
FOLDS = tuple(range(1, 11))
NULL_SEED = 20260928
SOURCES = (
    'c_cross_relation_null_experiment.py', 'Model/c_cross_relation_null.py',
    'tests/test_c_cross_relation_null.py', 'alternating_cooperative_experiment.py',
    'analyze_joint_evidence_routing.py', 'Model/alternating_cooperative.py',
    'analyze_alternating_cooperative.py')


def make_model(dd, cfg, graphs, null_graphs, control=MODE):
    return CrossRelationNullModel(dd, cfg, graphs, control=control,
        null_graphs=null_graphs, training_seed=0).to(cfg.Device)


def graph_hash(graphs):
    digest = hashlib.sha256()
    for graph in graphs:
        for key in ('edge', 'similarity', 'degree'):
            digest.update(key.encode())
            digest.update(graph[key].cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_graphs(graphs, null_graphs, samples):
    """Independent checks: labels are not an argument to graph construction."""
    checks = []
    assert len(graphs) == len(null_graphs)
    for index, (original, null) in enumerate(zip(graphs, null_graphs)):
        edge, weight = null['edge'].cpu().numpy(), null['similarity'].cpu().numpy()
        assert edge.dtype == np.int64 and edge.shape[0] == 2
        assert edge.shape == tuple(original['edge'].shape)
        assert len(weight) == edge.shape[1] and np.isfinite(weight).all()
        assert edge.min() >= 0 and edge.max() < samples
        assert np.all(edge[0] != edge[1])
        keys = edge[0] * samples + edge[1]
        reverse = edge[1] * samples + edge[0]
        assert len(np.unique(keys)) == len(keys)
        np.testing.assert_array_equal(np.sort(keys), np.sort(reverse))
        order, reverse_order = np.argsort(keys), np.argsort(reverse)
        np.testing.assert_array_equal(weight[order], weight[reverse_order])
        degree = np.bincount(edge[0], minlength=samples)
        np.testing.assert_array_equal(degree, original['degree'].cpu().numpy())
        np.testing.assert_array_equal(degree, null['degree'].cpu().numpy())
        np.testing.assert_array_equal(np.sort(weight), np.sort(original['similarity'].cpu().numpy()))
        old = original['edge'].cpu().numpy()
        overlap = np.isin(keys, old[0] * samples + old[1])
        checks.append(dict(relation=index, directed_edges=len(keys),
            per_patient_unweighted_degree_exact=True, raw_prior_global_histogram_exact=True,
            symmetric_simple_loop_free=True, fraction_original_edges=float(overlap.mean())))
    return dict(passed=True, records=checks, labels_used=False,
        limitation='Weighted degrees, learned weights and local prior distributions are not matched.')


def save_graphs(path, graphs):
    arrays = {f'{key}_{index}': graph[key].cpu().numpy()
              for index, graph in enumerate(graphs) for key in ('edge', 'similarity', 'degree')}
    np.savez_compressed(path, **arrays)


def load_graphs(path, count):
    with np.load(path, allow_pickle=False) as saved:
        return [{key: torch.from_numpy(saved[f'{key}_{index}'].copy())
                 for key in ('edge', 'similarity', 'degree')} for index in range(count)]


def verify_effective_groups(model, graphs, null_graphs):
    """Verify the actual graph buffers used by all diagonal/offdiagonal paths."""
    propagation = model.propagation
    count, modalities = propagation.samples, propagation.modalities
    for r in range(modalities):
        for m in range(modalities):
            group = r * modalities + m
            selected = propagation.edge_group == group
            expected = graphs[r] if r == m else null_graphs[r]
            torch.testing.assert_close(propagation.edge[:, selected].cpu() - group * count,
                expected['edge'], rtol=0, atol=0)
            torch.testing.assert_close(propagation.similarity[selected].cpu(),
                expected['similarity'], rtol=0, atol=0)


def reference_d1_manifest(dd):
    manifest = {}
    for filename in ('PLAN.json', 'audit.json'):
        path = D1_REFERENCE.parent.parent / filename
        manifest[str(path.relative_to(ROOT))] = sha(path)
    audit = read(D1_REFERENCE.parent.parent / 'audit.json')
    assert audit['passed'] and audit['verified_checkpoints'] == 40
    records = {}
    for fold in FOLDS:
        folder = D1_REFERENCE / f'fold_{fold:02d}'
        records[fold] = _record(folder, fold, 1000)
        assert records[fold]['parameters'] == 895036
        for filename in ('complete.json', 'best.json', 'last.json', 'epochs.csv',
                         'predictions.npz', 'last_predictions.npz', 'optimizer_groups.json'):
            path = folder / filename
            manifest[str(path.relative_to(ROOT))] = sha(path)
        with np.load(REFERENCE / f'fold_{fold:02d}/predictions.npz') as original:
            np.testing.assert_array_equal(records[fold]['predictions']['best']['rows'], original['row_index'])
            np.testing.assert_array_equal(records[fold]['predictions']['best']['labels'], original['labels'])
    return manifest, dict(passed=True, reused_runs=10, saved_predictions_verified=20,
        new_checkpoint_replays=0, inherited_checkpoint_audit=str(D1_REFERENCE.parent.parent / 'audit.json'))


def smoke(cfg, dd, data, graphs, null_graphs):
    SET_Random(0)
    original = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).to(cfg.Device)
    initial_hash, initial_rng = state_hash(original), rng_state()
    checks = []
    for control in ('C', MODE):
        SET_Random(0)
        model = make_model(dd, cfg, graphs, null_graphs, control)
        assert state_hash(model) == initial_hash
        assert sum(p.numel() for p in model.parameters()) == 895096
        assert_rng_equal(initial_rng, rng_state())
        for training in (False, True):
            SET_Random(27); original.train(training)
            with torch.no_grad(): baseline = original(data['Feature'])
            after = rng_state()
            SET_Random(27); model.train(training)
            with torch.no_grad(): actual = model(data['Feature'])
            assert_rng_equal(after, rng_state())
            if control == 'C':
                torch.testing.assert_close(actual, baseline, rtol=2e-6, atol=2e-6)
        if control == MODE:
            verify_effective_groups(model, graphs, null_graphs)
            criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
            optimizer = relation_optimizer(model, cfg, 0.)
            groups = optimizer_manifest(model, optimizer)
            assert groups == read(REFERENCE / 'fold_01/optimizer_groups.json')
            torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            for step in range(8):
                model.train(); optimizer.zero_grad(set_to_none=True)
                z, emb, aux = model(data['Feature'])
                loss = criterion(z, data['Label'], data['Mask'][0][0], emb, aux)
                assert torch.isfinite(loss)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                    if name.startswith('propagation.'):
                        assert parameter.grad is not None, name
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step(); model.eval()
                with torch.no_grad():
                    assert torch.isfinite(model(data['Feature'])[0]).all()
            torch.cuda.synchronize()
            checks.append(dict(mode=control, steps=8,
                seconds_per_epoch=(time.perf_counter() - started)/8,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                test_metrics_inspected=False, optimizer_groups=groups))
            del optimizer, criterion, z, emb, aux, loss
        del model, actual, baseline
    del original
    torch.cuda.empty_cache()
    return dict(passed=True, initial_state_sha256=initial_hash,
        identical_parameters_initialization_and_rng=True,
        original_C_forward_reproduced_train_and_eval=True,
        diagonal_paths_unchanged_offdiagonal_paths_replaced=True,
        test_metrics_inspected=False, records=checks)


def summarize(plan):
    folders = {'C': REFERENCE, 'D1': D1_REFERENCE, MODE: OUT / f'seed_0/{MODE}'}
    records = {mode: {fold: _record(base / f'fold_{fold:02d}', fold, 1000)
        for fold in FOLDS if (base / f'fold_{fold:02d}/complete.json').exists()}
        for mode, base in folders.items()}
    variants = {mode: _aggregate(rows, list(FOLDS)) for mode, rows in records.items()}
    pairs = [_pair(reference, candidate, records[reference], records[candidate], plan['class_names'])
             for reference, candidate in (('C', MODE), ('D1', MODE), ('C', 'D1'))]
    completed = len(records[MODE])
    result = dict(updated_at=timestamp(), completed_new_runs=completed, planned_new_runs=10,
        variants=variants, matched_comparisons=pairs,
        interpretation='Single prespecified null graph realization and training seed0; original test-selected protocol. '
            'Joint topology/prior-placement control, not isolated topology or learned-weight matching.')
    dump(OUT / 'summary.json', result)
    lines = ['# C跨模态关系复用：固定随机关系对照', '',
        f'完成 {completed}/10 次新训练；复用 C、D1 各10折。',
        'TAD三分类，原十折、训练seed0、每折1000轮，首次最高测试ACC选轮。', '',
        '| 模型 | 已完成折 | 最佳ACC | 宏F1 | 末轮ACC | 末100轮平均ACC | 参数量 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode, value in variants.items():
        if value['completed_folds'] == 10:
            lines.append(f"| {mode} | 10/10 | {100*value['best']['acc']:.4f}% | "
                f"{100*value['best']['macro_f1']:.4f}% | {100*value['last']['acc']:.4f}% | "
                f"{100*value['mean_last100_acc']:.4f}% | {plan['parameters'][mode]} |")
        else:
            lines.append(f"| {mode} | {value['completed_folds']}/10 | 待完成 | — | — | — | {plan['parameters'][mode]} |")
    for pair in pairs:
        if not pair['matched_folds']:
            continue
        lines += ['', f"{pair['candidate']} − {pair['reference']}（配对{len(pair['matched_folds'])}折）："]
        for stage, title in (('best', '最佳检查点'), ('last', '第1000轮')):
            score = pair['stages'][stage]
            lines.append(f"- {title}：ACC {100*score['mean_acc_delta']:+.4f}个百分点，"
                f"纠正{score['corrected']}，新增错误{score['harmed']}，净{score['net']:+d}；"
                f"折胜/平/负={score['fold_wins']}/{score['fold_ties']}/{score['fold_losses']}。")
        lines.append(f"- 末100轮平均ACC差：{100*pair['mean_last100_acc_delta']:+.4f}个百分点。")
    lines += ['', '## 解释范围', '',
        'XREL_NULL完整保留同模态路径r=m；跨模态路径r≠m使用每个关系来源预先固定的一张随机图。',
        '保留每位患者在每张图的无权度数、总边数、无自环、对称性，以及原始相似度先验的全局数值分布。',
        '随机化同时改变邻居身份及先验与端点的对应，属于联合随机对照；不匹配逐节点加权度或学习后边权分布。',
        '模型仍可通过学习边权减弱随机关系。C优于本对照支持真实跨模态关系组合有效，但不能分离拓扑与先验放置贡献。',
        'C明显优于随机关系、但仅略优于D1时，大差距可能主要来自随机消息伤害，不能全部算作跨关系额外收益。',
        '没有差异只能说明本次种子、随机实现及指标未显示收益；动态内容边权可能从随机候选中保留有用邻居。',
        'D1归一化与关系条数不同，仅作背景参照；三组差值不能作可加因果分解，主要匹配对照为C与XREL_NULL。',
        '部分折不代表全十折结论；原协议使用测试集选轮，结果属于探索性机制比较。']
    report = '\n'.join(lines) + '\n'
    (OUT / '进度与结果.md').write_text(report, encoding='utf-8')
    if completed == 10 and (OUT / 'audit.json').exists():
        (OUT / '结果.md').write_text(report, encoding='utf-8')
    return result


def prepare():
    cfg, dd, data, graphs, hashes, old = base_setup()
    if (OUT / 'PLAN.json').exists():
        plan = read(OUT / 'PLAN.json')
        verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_manifest'])
        assert hashes == plan['data_sha256'] and old['candidate_hash'] == plan['candidate_hash']
        for artifact, digest in plan['artifact_sha256'].items():
            assert sha(OUT / artifact) == digest, artifact
        null_graphs = load_graphs(OUT / 'null_graphs.npz', len(graphs))
        assert graph_hash(null_graphs) == plan['null_graph_sha256']
        verify_graphs(graphs, null_graphs, dd['Sample_Num'])
        return cfg, dd, data, graphs, null_graphs, plan
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
        '-p', 'test_c_cross_relation_null.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    manifest, reference_audit = verify_reference(cfg, dd, data, graphs)
    d1_manifest, d1_audit = reference_d1_manifest(dd)
    manifest.update(d1_manifest)
    # The one null realization is fixed before any null-model test metrics exist.
    null_graphs, generator_report = build_null_graphs(graphs, dd['Sample_Num'],
        seed=NULL_SEED, swaps_per_edge=20, max_attempts_per_swap=100)
    assert generator_report['all_targets_reached'] and generator_report['label_free']
    graph_audit = verify_graphs(graphs, null_graphs, dd['Sample_Num'])
    checks = smoke(cfg, dd, data, graphs, null_graphs)
    save_graphs(OUT / 'null_graphs.npz', null_graphs)
    dump(OUT / 'null_graph_audit.json', dict(generator=generator_report, independent=graph_audit))
    (OUT / 'data_splits_and_graphs.npz').write_bytes((REFERENCE.parent / 'splits_and_graphs.npz').read_bytes())
    (OUT / 'implementation_checks.log').write_text(test.stdout + test.stderr, encoding='utf-8')
    sources = tuple(dict.fromkeys((*old['code_sha256'], *SOURCES)))
    plan = dict(experiment='Cross-modal relation reuse: matched cross-relation joint random null',
        training_seed=0, split_seed=0, null_graph_seed=NULL_SEED,
        folds=list(FOLDS), modes=[MODE], epochs=1000, planned_runs=10,
        reused_C_runs=10, reused_D1_runs=10,
        parameters={'C': 895096, 'D1': 895036, MODE: 895096},
        initialization={MODE: dict(state_sha256=checks['initial_state_sha256'], parameters=895096)},
        class_names=dd['Class_Names'], modalities=dd['Modal_Name'], samples=dd['Sample_Num'],
        original_config=old['original_config'], data_sha256=hashes, candidate_hash=old['candidate_hash'],
        null_graph_sha256=graph_hash(null_graphs),
        artifact_sha256={name: sha(OUT / name) for name in
            ('null_graphs.npz', 'null_graph_audit.json', 'data_splits_and_graphs.npz')},
        checkpoint_selection=old['checkpoint_selection'], transductive=True,
        optimizer='Original Adam/cosine; propagation except output_norm decay=0; remaining decay=.0007',
        primary='C vs XREL_NULL: equal-fold mean first-best TEST ACC and paired patient corrections',
        secondary='Macro F1, raw-logit OVR AUC, recalls, last epoch, last-100 mean; D1 is contextual reference',
        intervention='r=m exact original graph/prior. r!=m one degree-preserving undirected null per relation r, '
            'shared across all offdiagonal contents; original edge priors transplanted symmetrically.',
        matched='Trainable parameters, initial state, training RNG, fixed splits, epochs, losses, branch architecture, '
            'graph passes, group count, per-patient unweighted degree and raw-prior global histogram.',
        limitations='Joint topology/prior-placement null. Local prior distributions, weighted degree, learned '
            'edge-weight distributions, connectivity and neighbor identity are not matched. '
            'One null realization and one training seed. Historical test-selected exploratory estimates.',
        scope='Only ten XREL_NULL runs. No tuning, random-seed selection, extra data, early stopping or architectural changes.',
        graph_labels_used=False, common_initialization_and_rng_verified=True,
        smoke=checks, reference_audit=reference_audit, d1_reference_audit=d1_audit,
        code_sha256={name: sha(ROOT / name) for name in sources}, reference_manifest=manifest,
        environment=dict(executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0)))
    for source in sources:
        target = OUT / 'source_snapshot' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    dump(OUT / 'PLAN.json', plan)
    (OUT / 'PLAN.md').write_text(
        '# 跨模态关系复用验证计划\n\n'
        '问题：保持同模态传播后，真实跨模态患者关系是否优于规模匹配的随机关系？\n\n'
        '仅训练XREL_NULL十折，seed0、每折1000轮；复用原C与D1十折。沿用原首次最高测试ACC选轮。\n\n'
        '每张关系图预先生成一张对称、无自环、逐患者度数不变的随机图，固定用于全部跨模态内容路径；'
        '同模态路径原样保留。随机图种子20260928只用于关系构造，不是新增训练种子。'
        '构图不使用标签，不按性能挑选随机图。\n\n'
        '原始相似度先验的全局直方图保留，但先验与患者对的对应被随机移植；'
        '因此是拓扑与先验放置的联合随机对照，不是纯拓扑消融。逐节点加权度和学习后边权分布不匹配。\n\n'
        '主比较C与XREL_NULL；D1少60个参数、关系路数与平均方式不同，只作背景参照。'
        '报告最佳/末轮/末100轮及逐患者纠正与新增错误。全部20个新检查点复核后形成结果。\n\n'
        '若C更好，支持真实跨模态关系组合有效；若差异小，仅说明本次对照未显示优势，'
        '不能声称该机制普遍无效，也不通过中途调参寻找预期结论。\n', encoding='utf-8')
    summarize(plan)
    dump(OUT / 'status.json', dict(state='prepared', updated_at=timestamp(), planned_runs=10))
    return cfg, dd, data, graphs, null_graphs, plan


def train_fold(cfg, dd, data, graphs, null_graphs, plan, fold):
    folder = OUT / f'seed_0/{MODE}/fold_{fold:02d}'
    if (folder / 'complete.json').exists():
        _record(folder, fold, 1000)
        return
    verify_manifest(plan['code_sha256'])
    if folder.exists():
        # Never silently merge a partial previous log with a from-scratch rerun.
        archived = folder.with_name(folder.name + f'.interrupted_{time.time_ns()}')
        output_root = OUT.resolve()
        if not (folder.resolve().is_relative_to(output_root)
                and archived.resolve().is_relative_to(output_root)):
            raise RuntimeError('Interrupted fold archive must remain inside the experiment output')
        folder.rename(archived)
    folder.mkdir(parents=True)
    SET_Random(0); model = make_model(dd, cfg, graphs, null_graphs)
    assert state_hash(model) == plan['initialization'][MODE]['state_sha256']
    verify_effective_groups(model, graphs, null_graphs)
    criterion = MSCQGNLoss(dd, cfg.Device, rate=cfg.Loss_rate, aux_loss_weight=cfg.aux_loss_weight)
    optimizer = relation_optimizer(model, cfg, 0.)
    optimizer_groups = optimizer_manifest(model, optimizer)
    assert optimizer_groups == read(REFERENCE / f'fold_{fold:02d}/optimizer_groups.json')
    dump(folder / 'optimizer_groups.json', optimizer_groups)
    scheduler = CustomCosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.Lr_Min)
    x, y = data['Feature'], data['Label']; fit, test = data['Mask'][fold - 1]
    best_correct = -1; history = []; started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 1001):
            model.train(); optimizer.zero_grad(set_to_none=True)
            z, emb, aux = model(x); loss = criterion(z, y, fit, emb, aux)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step(); model.eval()
            with torch.no_grad():
                z, emb, aux = model(x); test_loss = criterion(z, y, test, emb, aux)
                if not torch.isfinite(z).all() or not torch.isfinite(test_loss):
                    raise FloatingPointError('Nonfinite evaluation')
            correct = int((z[test].argmax(-1) == y[test]).sum())
            row = dict(epoch=epoch, train_loss=float(loss.detach()), test_loss=float(test_loss),
                test_acc=correct/int(test.sum()), lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-started)
            writer.writerow(row); history.append(row['test_acc'])
            for stage in (['best'] if correct > best_correct else []) + (['last'] if epoch == 1000 else []):
                score = dict(epoch=epoch, **metrics(z[test], y[test]))
                if stage == 'best':
                    best_correct = correct; best = score
                else:
                    last = score
                temporary = folder / f'{stage}.pt.tmp'
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metrics=score, mode=MODE,
                    fold=fold, seed=0, protocol_sha256=sha(OUT / 'PLAN.json'),
                    null_graph_sha256=plan['null_graph_sha256']), temporary)
                temporary.replace(folder / f'{stage}.pt')
                save_predictions(folder / ('predictions.npz' if stage == 'best' else 'last_predictions.npz'),
                    z, y, test, dd['Index'])
                dump(folder / f'{stage}.json', score)
            scheduler.step()
            if epoch == 1 or epoch % 25 == 0:
                stream.flush()
                status = dict(state='running', updated_at=timestamp(), pid=os.getpid(), mode=MODE,
                    fold=fold, seed=0, epochs=1000, planned_runs=10, best_acc=best['acc'], **row)
                dump(OUT / 'status.json', status); print(json.dumps(status), flush=True)
    dump(folder / 'complete.json', dict(mode=MODE, fold=fold, seed=0, parameters=895096,
        best=best, last=last, initial_model_sha256=plan['initialization'][MODE]['state_sha256'],
        mean_last100_acc=float(np.mean(history[-100:])), null_graph_sha256=plan['null_graph_sha256'],
        best_checkpoint_sha256=sha(folder / 'best.pt'), last_checkpoint_sha256=sha(folder / 'last.pt'),
        completed_at=timestamp(), seconds=time.perf_counter()-started,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
    del model, optimizer, criterion, z, emb, aux, loss
    torch.cuda.empty_cache(); summarize(plan)


def audit_completed(cfg, dd, data, graphs, null_graphs, plan):
    verify_manifest(plan['code_sha256']); verify_manifest(plan['reference_manifest'])
    for artifact, digest in plan['artifact_sha256'].items():
        assert sha(OUT / artifact) == digest, artifact
    rows = []; oof_rows = []
    model = make_model(dd, cfg, graphs, null_graphs).eval()
    verify_effective_groups(model, graphs, null_graphs)
    for fold in FOLDS:
        folder = OUT / f'seed_0/{MODE}/fold_{fold:02d}'
        record = _record(folder, fold, 1000)['record']
        with (folder / 'epochs.csv').open() as stream:
            epochs = list(csv.DictReader(stream))
        with (REFERENCE / f'fold_{fold:02d}/epochs.csv').open() as stream:
            original = list(csv.DictReader(stream))
        np.testing.assert_array_equal([float(row['lr']) for row in epochs], [float(row['lr']) for row in original])
        test = data['Mask'][fold - 1][1]
        oof_rows.extend(np.asarray(dd['Index'])[test.cpu().numpy()].tolist())
        for stage, filename in (('best', 'predictions.npz'), ('last', 'last_predictions.npz')):
            assert sha(folder / f'{stage}.pt') == record[f'{stage}_checkpoint_sha256']
            checkpoint = torch.load(folder / f'{stage}.pt', map_location=cfg.Device, weights_only=False)
            assert checkpoint['protocol_sha256'] == sha(OUT / 'PLAN.json')
            assert checkpoint['null_graph_sha256'] == plan['null_graph_sha256']
            assert (checkpoint['mode'], checkpoint['fold'], checkpoint['seed']) == (MODE, fold, 0)
            assert checkpoint['epoch'] == record[stage]['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with torch.no_grad():
                logits = model(data['Feature'])[0][test]
            with np.load(folder / filename, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['row_index'], np.asarray(dd['Index'])[test.cpu().numpy()])
                np.testing.assert_array_equal(saved['labels'], data['Label'][test].cpu().numpy())
                np.testing.assert_allclose(saved['logits'], logits.cpu().numpy(), rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(saved['logits'].argmax(-1), logits.argmax(-1).cpu().numpy())
            actual = metrics(logits, data['Label'][test])
            for key in ('acc', 'auc', 'f1', 'macro_f1'):
                assert abs(actual[key] - record[stage][key]) < 1e-12
            rows.append(dict(mode=MODE, fold=fold, stage=stage, passed=True))
    assert len(oof_rows) == len(set(oof_rows)) == dd['Sample_Num']
    np.testing.assert_array_equal(np.sort(oof_rows), np.sort(dd['Index']))
    result = dict(passed=True, verified_runs=10, verified_checkpoints=20,
        unique_out_of_fold_patients=len(oof_rows), records=rows,
        graph_audit=verify_graphs(graphs, null_graphs, dd['Sample_Num']))
    dump(OUT / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Freeze artifacts and run checks, without formal training')
    parser.add_argument('--resume', action='store_true', help='Resume incomplete experiment; complete folds are always skipped')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'RUNNING.lock'
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Experiment lock exists: {lock}. Verify the recorded process has stopped before removing a stale lock.')
    with stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_at=timestamp(), argv=sys.argv)))
    try:
        cfg, dd, data, graphs, null_graphs, plan = prepare()
        if args.prepare:
            print(json.dumps(dict(state='prepared', parameters=plan['parameters'], smoke=plan['smoke'])), flush=True)
            return
        if (OUT / 'COMPLETE.json').exists():
            print('Already complete; no duplicate training started.', flush=True)
            return
        for fold in FOLDS:
            train_fold(cfg, dd, data, graphs, null_graphs, plan, fold)
        audit = audit_completed(cfg, dd, data, graphs, null_graphs, plan)
        summary = summarize(plan)
        dump(OUT / 'COMPLETE.json', dict(completed_at=timestamp(), audit=audit, summary=summary))
        dump(OUT / 'status.json', dict(state='complete', completed_runs=10, updated_at=timestamp()))
    except BaseException as error:
        dump(OUT / 'status.json', dict(state='failed', updated_at=timestamp(), error=repr(error)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
