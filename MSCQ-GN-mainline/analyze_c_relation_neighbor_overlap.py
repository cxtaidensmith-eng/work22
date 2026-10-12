"""Analyze actual fixed candidate graphs for Framework C; no training/inference.

Neighbor sets use final symmetric support, not the directed top-k selections.
The cohort is analyzed once; folds are not treated as ten graph replicates.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from relation_experiment import ROOT, dump, setup, sha, timestamp


REFERENCE = ROOT / 'Result/relation_mechanism_nodecay/seed_0'
PREVIOUS = ROOT / 'Result/c_edge_content_weights_20261009'
DEFAULT_OUTPUT = ROOT / 'Result/c_relation_neighbor_overlap_20261010'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd_descriptive=float(values.std(ddof=0)), minimum=float(values.min()),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                maximum=float(values.max()))


def validate_inputs(cfg, dd, data, graphs, paths):
    historical = read(REFERENCE / 'PROTOCOL.json')
    prior = read(PREVIOUS / 'PLAN.json')
    assert cfg.DATA_SET == 'TADPOLE' and cfg.Task == 'AD_CN_SMCI'
    assert dd['Sample_Num'] == 598 and len(graphs) == len(dd['Modal_Name']) == 6
    assert cfg.seed == historical['split_seed'] == 0
    assert list(dd['Modal_Name']) == prior['modalities']
    for key, value in historical['original_config'].items():
        assert getattr(cfg, key) == value, key
    for name, digest in historical['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert {Path(p).name: sha(p) for p in paths} == historical['data_sha256']
    graph_hash = hashlib.sha256(b''.join(g['edge'].numpy().tobytes() for g in graphs)).hexdigest()
    assert graph_hash == historical['candidate_hash'] == prior['candidate_hash']
    masks = np.stack([test.cpu().numpy() for _, test in data['Mask']])
    assert masks.shape == (10, 598) and np.all(masks.sum(0) == 1)
    with np.load(REFERENCE / 'splits_and_graphs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_row'], dd['Index'])
        np.testing.assert_array_equal(saved['test_masks'], masks)
        for m, graph in enumerate(graphs):
            np.testing.assert_array_equal(saved[f'edge_{m}'], graph['edge'].numpy())
            np.testing.assert_array_equal(saved[f'neighbors_{m}'], graph['neighbors'].numpy())
    # The prior weight experiment selected receiving edges by held-out fold.
    # Reassemble its ten partitions to prove that both experiments use one graph.
    previous_edges = [[] for _ in graphs]
    for fold in range(1, 11):
        with np.load(PREVIOUS / f'fold_{fold:02d}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['modality_names'], np.asarray(dd['Modal_Name']))
            for m in range(len(graphs)):
                receiver, sender = saved[f'r{m}_receiver'], saved[f'r{m}_sender']
                assert np.all(masks[fold - 1, receiver])
                previous_edges[m].append(receiver.astype(np.int64) * 598 + sender)
    for m, pieces in enumerate(previous_edges):
        actual = np.sort(np.concatenate(pieces))
        edge = graphs[m]['edge'].numpy()
        expected = np.sort(edge[0].astype(np.int64) * 598 + edge[1])
        np.testing.assert_array_equal(actual, expected)
        assert len(np.unique(actual)) == len(actual)
    sources = [Path(__file__).resolve(), REFERENCE / 'PROTOCOL.json',
               REFERENCE / 'splits_and_graphs.npz', PREVIOUS / 'PLAN.json']
    sources += [ROOT / name for name in historical['code_sha256']]
    sources += [Path(p) for p in paths]
    sources += [PREVIOUS / f'fold_{f:02d}.npz' for f in range(1, 11)]
    sources = list(dict.fromkeys(sources))
    return historical, graph_hash, {str(p): sha(p) for p in sources}


@lru_cache(None)
def chance_jaccard(population, degree_a, degree_b):
    """Exact E[J] for independent degree-matched uniform neighbor sets.

    This is a set-size reference, not a null graph or a significance test.
    E[K/(d_a+d_b-K)] is not replaced by a ratio involving E[K].
    """
    low = max(0, degree_a + degree_b - population)
    high = min(degree_a, degree_b)
    common = np.arange(low, high + 1)
    probability = hypergeom.pmf(common, population, degree_a, degree_b)
    assert np.isclose(probability.sum(), 1, atol=1e-10)
    return float(np.sum(probability * common / (degree_a + degree_b - common)))


def analyze(output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    cfg, dd, data, graphs, paths = setup(cuda=-1, k=10)
    historical, graph_hash, input_hashes = validate_inputs(cfg, dd, data, graphs, paths)
    names = list(dd['Modal_Name'])
    n, m = dd['Sample_Num'], len(names)
    output.mkdir(parents=True)
    plan = dict(created_at=timestamp(), experiment='Framework C relation-source neighbor overlap',
                dataset='TADPOLE AD/CN/SMCI', sample_count=n, modalities=names, candidate_k=10,
                support='Processed raw modality cosine top-10; union with reverse edges; exclude self',
                candidate_hash=graph_hash, statistical_unit='Each patient has equal weight; one static cohort graph set',
                jaccard='|N_r(i) intersect N_s(i)| / |N_r(i) union N_s(i)|',
                representative_selection='MRI/PET Jaccard closest to cohort median; then union degree closest to cohort median; then analysis row order',
                example_uses_diagnosis=False, no_training=True, no_checkpoint_inference=True,
                chance_reference='Exact E[J] of independent uniform neighbor sets with the two actual patient degrees among 597 possible neighbors; not a random symmetric graph or classification baseline',
                scope='Distinct neighborhoods do not prove useful complementarity, classification gains or clinical causality',
                input_sha256=input_hashes)
    dump(output / 'PLAN.json', plan)
    adjacency = np.zeros((m, n, n), dtype=bool)
    selections = np.zeros_like(adjacency)
    for relation, graph in enumerate(graphs):
        edge = graph['edge'].numpy()
        adjacency[relation, edge[0], edge[1]] = True
        top = graph['neighbors'].numpy()
        selections[relation, np.arange(n)[:, None], top] = True
        np.testing.assert_array_equal(adjacency[relation], selections[relation] | selections[relation].T)
        assert not np.any(np.diag(adjacency[relation]))
        assert np.all(adjacency[relation] == adjacency[relation].T)
        assert np.all(selections[relation].sum(1) == 10)
        np.testing.assert_array_equal(adjacency[relation].sum(1), graph['degree'].numpy())
    degree = adjacency.sum(2).astype(np.int32)
    jaccard = np.empty((m, m, n), dtype=np.float64)
    intersection = np.empty((m, m, n), dtype=np.int32)
    union = np.empty_like(intersection)
    chance = np.empty_like(jaccard)
    for a in range(m):
        for b in range(m):
            intersection[a, b] = (adjacency[a] & adjacency[b]).sum(1)
            union[a, b] = (adjacency[a] | adjacency[b]).sum(1)
            jaccard[a, b] = intersection[a, b] / union[a, b]
            if a == b:
                chance[a, b] = np.nan  # identity is not an independent-set draw
            else:
                chance[a, b] = [chance_jaccard(n - 1, int(da), int(db))
                                for da, db in zip(degree[a], degree[b])]
    assert np.isfinite(jaccard).all() and np.all((jaccard >= 0) & (jaccard <= 1))
    np.testing.assert_array_equal(jaccard, jaccard.transpose(1, 0, 2))
    assert np.all(jaccard[np.arange(m), np.arange(m)] == 1)
    mri, pet = names.index('MRI'), names.index('PET')
    main = jaccard[mri, pet]
    median = np.median(main)
    union_median = np.median(union[mri, pet])
    order = np.lexsort((np.arange(n), np.abs(union[mri, pet] - union_median), np.abs(main - median)))
    example = int(order[0])
    example_union = np.flatnonzero(adjacency[:, example].any(0))
    example_membership = adjacency[:, example, example_union]
    pair_indices = np.asarray(list(itertools.combinations(range(m), 2)), dtype=np.int32)
    mean_matrix = jaccard.mean(2)
    pairs = []
    for a, b in pair_indices:
        pairs.append(dict(modality_a=names[a], modality_b=names[b],
                          jaccard=describe(jaccard[a, b]),
                          common_neighbor_count=describe(intersection[a, b]),
                          only_a_count=describe(degree[a] - intersection[a, b]),
                          only_b_count=describe(degree[b] - intersection[a, b]),
                          union_neighbor_count=describe(union[a, b]),
                          identical_neighbor_patients=int(np.sum(jaccard[a, b] == 1)),
                          disjoint_neighbor_patients=int(np.sum(intersection[a, b] == 0)),
                          exact_chance_jaccard_mean=float(chance[a, b].mean()),
                          global_undirected_edge_jaccard=float(intersection[a, b].sum() / union[a, b].sum())))
    other_support_count = adjacency.sum(0)
    graph_stats = []
    for r, name in enumerate(names):
        unique = (adjacency[r] & (other_support_count == 1)).sum(1)
        graph_stats.append(dict(modality=name, actual_degree=describe(degree[r]),
                                directed_edge_records=int(degree[r].sum()),
                                undirected_edges=int(degree[r].sum() // 2),
                                neighbors_unique_to_this_modality=describe(unique)))
    example_details = dict(anonymous_target='Target patient', analysis_row_ordinal=example,
                           selection_criterion=plan['representative_selection'],
                           mri_pet_jaccard=float(main[example]), cohort_mri_pet_median=float(median),
                           mri_degree=int(degree[mri, example]), pet_degree=int(degree[pet, example]),
                           shared=int(intersection[mri, pet, example]),
                           mri_only=int(degree[mri, example] - intersection[mri, pet, example]),
                           pet_only=int(degree[pet, example] - intersection[mri, pet, example]),
                           union=int(union[mri, pet, example]),
                           all_six_union=len(example_union),
                           labels='N01... anonymous cohort-row aliases, shared across all panels; no clinical IDs/diagnosis')
    arrays = dict(modality_names=np.asarray(names, dtype='U32'), adjacency=adjacency,
                  directed_topk_support=selections, degree=degree, jaccard=jaccard,
                  intersection=intersection, union=union, mean_jaccard=mean_matrix,
                  chance_jaccard=chance, pair_indices=pair_indices,
                  example_patient=np.asarray(example), example_union=example_union,
                  example_membership=example_membership, mri_index=np.asarray(mri), pet_index=np.asarray(pet))
    np.savez_compressed(output / 'analysis.npz', **arrays)
    summary = dict(completed_at=timestamp(), sample_count=n, modalities=names,
                   candidate_hash=graph_hash, statistical_unit=plan['statistical_unit'],
                   mean_jaccard_matrix=mean_matrix.tolist(), pairs=pairs, graph_statistics=graph_stats,
                   example=example_details, audit=dict(historical_graphs_exact_match=True,
                     previous_weight_experiment_graphs_exact_match=True, symmetric_support=True,
                     self_edges=False, pre_symmetrization_degree=10, diagnosis_used=False),
                   scope=plan['scope'])
    dump(output / 'summary.json', summary)
    for name, digest in input_hashes.items():
        assert sha(name) == digest, f'Input changed during analysis: {name}'
    outputs = [output / name for name in ('analysis.npz', 'PLAN.json', 'summary.json')]
    dump(output / 'COMPLETE.json', dict(completed_at=timestamp(), state='complete',
            output_sha256={p.name: sha(p) for p in outputs}, input_sha256=input_hashes))
    print(json.dumps(dict(output=str(output), main_mri_pet=pairs[0], example=example_details,
                          graph_statistics=graph_stats, complete=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(args.output.resolve())
