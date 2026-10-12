"""Label-free, degree-preserving null for C's off-diagonal relation paths.

The model, forward pass, parameter budget, and training RNG are unchanged.
Only r != m graph buffers are replaced.  The r == m paths retain the exact
original edges and prior similarities.  A single null graph for each relation
source r is reused by every off-diagonal content path.

Randomization preserves unweighted degree for each patient and the global
undirected prior-similarity multiset.  It does NOT preserve learned edge weights,
per-patient weighted degree, connectivity, or each patient's prior histogram.
The swap process is a bounded randomization, not a claim of uniform sampling.
"""
import hashlib

import numpy as np
import torch

from .relation_sheaf import RelationSheafModel


def _tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_undirected(graph, samples):
    """Return sorted undirected edges and exact forward/reverse-shared priors."""
    edge = graph['edge'].detach().cpu()
    similarity = graph['similarity'].detach().cpu()
    if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError('Expected edge indices with int64 shape [2, E]')
    if similarity.ndim != 1 or len(similarity) != edge.shape[1]:
        raise ValueError('Expected one prior similarity per directed edge')
    if not similarity.is_floating_point() or not torch.isfinite(similarity).all():
        raise ValueError('Prior similarities must be finite floating point values')
    if samples < 2 or edge.numel() == 0:
        raise ValueError('Require at least two patients and a nonempty graph')
    if int(edge.min()) < 0 or int(edge.max()) >= samples:
        raise ValueError('Edge index outside the patient cohort')
    if bool((edge[0] == edge[1]).any()):
        raise ValueError('Self loops are not permitted')
    keys = edge[0] * samples + edge[1]
    sorted_keys, order = keys.sort()
    if len(torch.unique_consecutive(sorted_keys)) != len(sorted_keys):
        raise ValueError('Duplicate directed edges are not permitted')
    reverse_keys = edge[1] * samples + edge[0]
    locations = torch.searchsorted(sorted_keys, reverse_keys)
    if bool((locations >= len(sorted_keys)).any()) or not torch.equal(
            sorted_keys[locations], reverse_keys):
        raise ValueError('Every edge must have its reverse edge')
    if not torch.equal(similarity[order[locations]], similarity):
        raise ValueError('Forward and reverse priors must be exactly equal')
    keep = edge[0] < edge[1]
    undirected = edge[:, keep]
    priors = similarity[keep]
    indices = (undirected[0] * samples + undirected[1]).argsort()
    return undirected[:, indices].T.numpy().copy(), priors[indices].clone()


def _directed_graph(undirected, priors, samples):
    edge = torch.from_numpy(np.asarray(undirected, dtype=np.int64).T.copy())
    edge = torch.cat((edge, edge.flip(0)), dim=1)
    similarity = torch.cat((priors, priors))
    order = (edge[0] * samples + edge[1]).argsort()
    edge = edge[:, order].contiguous()
    return dict(edge=edge, similarity=similarity[order].contiguous(),
                degree=torch.bincount(edge[0], minlength=samples))


def _connectivity(edges, samples):
    parent = list(range(samples))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        parent[root(int(left))] = root(int(right))
    sizes = {}
    for node in range(samples):
        representative = root(node)
        sizes[representative] = sizes.get(representative, 0) + 1
    components = sorted(sizes.values(), reverse=True)
    return dict(component_count=len(components), component_sizes=components,
                largest_component_fraction=components[0] / samples,
                isolated_patients=sum(size == 1 for size in components))


def _cross_relation_overlap(edge_sets):
    return dict(
        undirected_intersection_count=[
            [len(left.intersection(right)) for right in edge_sets] for left in edge_sets],
        undirected_jaccard=[
            [len(left.intersection(right)) / len(left.union(right)) for right in edge_sets]
            for left in edge_sets])


def build_null_graphs(graphs, samples, seed=20260928, swaps_per_edge=20,
                      max_attempts_per_swap=100):
    """Construct independent degree-preserving nulls; no labels are accepted.

    Each relation uses numpy.default_rng(seed + relation_index), which does not
    touch the global NumPy, Python, CPU torch, or CUDA random streams.  Each
    successful swap replaces two disjoint undirected edges by two absent edges.
    A target may be unreachable for constrained graphs (e.g. complete graphs);
    the bounded result explicitly reports target_reached=False in that case.
    Production callers must inspect this status before running an experiment.
    """
    if not isinstance(swaps_per_edge, int) or swaps_per_edge < 0:
        raise ValueError('swaps_per_edge must be a nonnegative integer')
    if not isinstance(max_attempts_per_swap, int) or max_attempts_per_swap < 1:
        raise ValueError('max_attempts_per_swap must be a positive integer')
    if not graphs:
        raise ValueError('At least one relation graph is required')
    null_graphs, reports = [], []
    original_sets, null_sets = [], []
    for relation, graph in enumerate(graphs):
        original, original_priors = _validated_undirected(graph, samples)
        rng = np.random.default_rng(seed + relation)
        rewired = original.copy()
        original_set = set(map(tuple, original.tolist()))
        edge_set = set(original_set)
        count = len(rewired)
        target = swaps_per_edge * count
        limit = max_attempts_per_swap * target
        attempts = successful = 0
        while successful < target and attempts < limit and count >= 2:
            attempts += 1
            first, second = rng.integers(count, size=2).tolist()
            if first == second:
                continue
            a, b = rewired[first].tolist()
            c, d = rewired[second].tolist()
            if len({a, b, c, d}) != 4:
                continue
            if int(rng.integers(2)):
                c, d = d, c
            left = (min(a, c), max(a, c))
            right = (min(b, d), max(b, d))
            if left in edge_set or right in edge_set:
                continue
            edge_set.remove((a, b))
            edge_set.remove(tuple(rewired[second].tolist()))
            edge_set.add(left)
            edge_set.add(right)
            rewired[first] = left
            rewired[second] = right
            successful += 1
        # Assign the original global prior multiset to the randomized support.
        # This intentionally destroys the prior's correspondence to patient pairs.
        sorted_edges = np.asarray(sorted(edge_set), dtype=np.int64)
        shuffled_priors = original_priors[torch.from_numpy(rng.permutation(count))]
        null_graph = _directed_graph(sorted_edges, shuffled_priors, samples)
        validated_edges, validated_priors = _validated_undirected(null_graph, samples)
        original_degree = torch.bincount(graph['edge'][0].cpu(), minlength=samples)
        degree_exact = torch.equal(original_degree, null_graph['degree'])
        prior_exact = torch.equal(original_priors.sort().values,
                                  validated_priors.sort().values)
        if not degree_exact or not prior_exact or len(validated_edges) != count:
            raise RuntimeError('Null graph invariant failed')
        intersection = len(original_set.intersection(edge_set))
        union = len(original_set.union(edge_set))
        reports.append(dict(
            relation_index=relation, seed=int(seed + relation), samples=int(samples),
            undirected_edge_count=count, directed_edge_count=2 * count,
            target_swaps=target, successful_swaps=successful, attempts=attempts,
            attempt_limit=limit, target_reached=successful == target,
            degree_exact=degree_exact, prior_histogram_exact=prior_exact,
            symmetric=True, no_self_loops=True, no_duplicate_edges=True,
            edge_overlap_count=intersection, edge_overlap_fraction=intersection / count,
            edge_jaccard=intersection / union,
            degree_min=int(original_degree.min()), degree_max=int(original_degree.max()),
            degree_mean=float(original_degree.float().mean()),
            original_connectivity=_connectivity(original, samples),
            null_connectivity=_connectivity(sorted_edges, samples),
            original_edge_sha256=_tensor_hash(graph['edge']),
            original_similarity_sha256=_tensor_hash(graph['similarity']),
            null_edge_sha256=_tensor_hash(null_graph['edge']),
            null_similarity_sha256=_tensor_hash(null_graph['similarity']),
            degree_sha256=_tensor_hash(original_degree),
            sorted_undirected_prior_sha256=_tensor_hash(original_priors.sort().values)))
        null_graphs.append(null_graph)
        original_sets.append(original_set)
        null_sets.append(edge_set)
    report = dict(
        method='simple_undirected_double_edge_swap_then_global_prior_permutation',
        seed=int(seed), swaps_per_edge=swaps_per_edge,
        max_attempts_per_swap=max_attempts_per_swap,
        all_targets_reached=all(item['target_reached'] for item in reports),
        label_free=True, preserves_learned_weights=False,
        preserves_per_patient_weighted_degree=False,
        original_cross_relation_overlap=_cross_relation_overlap(original_sets),
        null_cross_relation_overlap=_cross_relation_overlap(null_sets),
        relations=reports)
    return null_graphs, report


def apply_cross_relation_control(model, graphs, control='C', null_graphs=None):
    """Modify only the three nonpersistent propagation buffers, in place.

    Supplying null_graphs=graphs is an identity intervention and is useful for
    exact forward, gradient, and RNG equivalence tests.  Null generation is
    explicit so callers can save its report and reuse it across all folds.
    """
    if control not in ('C', 'XREL_NULL'):
        raise ValueError('control must be C or XREL_NULL')
    propagation = model.propagation
    if propagation.mode != 'content':
        raise ValueError('This control is defined for the current C content mode')
    modalities, samples = propagation.modalities, propagation.samples
    if len(graphs) != modalities:
        raise ValueError('One original graph per relation source is required')
    if control == 'C':
        return model
    if null_graphs is None or len(null_graphs) != modalities:
        raise ValueError('XREL_NULL requires one explicit null graph per relation source')
    edges, priors, groups = [], [], []
    for relation in range(modalities):
        original_edges, original_priors = _validated_undirected(graphs[relation], samples)
        null_edges, null_priors = _validated_undirected(null_graphs[relation], samples)
        original_degree = torch.bincount(graphs[relation]['edge'][0].cpu(), minlength=samples)
        null_degree = torch.bincount(null_graphs[relation]['edge'][0].cpu(), minlength=samples)
        if len(original_edges) != len(null_edges) or not torch.equal(original_degree, null_degree):
            raise ValueError('Null graph must preserve each patient degree and edge count')
        if not torch.equal(original_priors.sort().values, null_priors.sort().values):
            raise ValueError('Null graph must preserve the undirected prior multiset')
        for content in range(modalities):
            group = relation * modalities + content
            graph = graphs[relation] if relation == content else null_graphs[relation]
            edges.append(graph['edge'].to(propagation.edge.device) + group * samples)
            priors.append(graph['similarity'].to(propagation.similarity))
            groups.append(torch.full((graph['edge'].shape[1],), group,
                                     dtype=propagation.edge_group.dtype,
                                     device=propagation.edge_group.device))
    propagation.edge = torch.cat(edges, dim=1)
    propagation.similarity = torch.cat(priors)
    propagation.edge_group = torch.cat(groups)
    return model


class CrossRelationNullModel(RelationSheafModel):
    """Current C with an optional buffer-only off-diagonal relation null."""
    def __init__(self, dd, cfg, graphs, control='C', null_graphs=None, training_seed=None):
        super().__init__(dd, cfg, graphs, 'content', training_seed=training_seed)
        apply_cross_relation_control(self, graphs, control, null_graphs)
