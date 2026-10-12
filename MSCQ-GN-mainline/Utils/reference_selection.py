"""Rewiring and train-only cross-fitting helpers for reference selection."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold


def rewire_candidates(graphs, features, modal_indices, seed=7109, swaps_per_edge=10):
    """Undirected double-edge swaps preserve EACH node's degree in each graph.

    Similarities are recomputed on actual endpoints using the original raw
    modality cosine rule, not shuffled independently of their edges.
    """
    result, records = [], []
    x = features.detach().cpu()
    for m, (graph, indices) in enumerate(zip(graphs, modal_indices)):
        rng = np.random.default_rng(seed + m)
        pairs = [tuple(e) for e in graph['edge'].T.tolist() if e[0] < e[1]]
        original = set(pairs)
        edges = set(pairs)
        target, accepted, attempts = swaps_per_edge * len(pairs), 0, 0
        while accepted < target and attempts < max(1000, target * 50):
            attempts += 1
            aidx, bidx = rng.integers(len(pairs), size=2)
            a, b = pairs[aidx]
            c, d = pairs[bidx]
            if len({a, b, c, d}) != 4:
                continue
            if rng.integers(2):
                c, d = d, c
            first, second = tuple(sorted((a, d))), tuple(sorted((c, b)))
            if first in edges or second in edges:
                continue
            edges.remove(pairs[aidx]); edges.remove(pairs[bidx])
            edges.update((first, second))
            pairs[aidx], pairs[bidx] = first, second
            accepted += 1
        if accepted != target:
            raise RuntimeError('Degree-preserving rewiring did not mix as planned')
        directed = sorted(edges | {(j, i) for i, j in edges})
        edge = torch.tensor(directed, dtype=torch.long).T.contiguous()
        degree = torch.bincount(edge[0], minlength=len(x))
        if not torch.equal(degree, graph['degree']):
            raise AssertionError('Rewiring changed a node degree')
        z = F.normalize(x[:, indices], dim=-1)
        similarity = (z[edge[0]] * z[edge[1]]).sum(-1)
        result.append(dict(edge=edge, similarity=similarity, degree=degree))
        records.append(dict(modality=m, edges=len(directed), accepted_swaps=accepted,
                            overlap_fraction=len(edges & original)/len(original),
                            degree_preserved=True, seed=seed+m))
    return result, records


def inner_splits(labels, outer_fit, outer_test, seed, n_splits=2):
    """Outer test rows never occur in either inner supervised partition."""
    y = labels.detach().cpu().numpy()
    fit = outer_fit.detach().cpu().numpy().astype(bool)
    test = outer_test.detach().cpu().numpy().astype(bool)
    if np.any(fit & test):
        raise ValueError('Outer masks overlap')
    indices = np.flatnonzero(fit)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for a, b in splitter.split(np.zeros(len(indices)), y[indices]):
        train = np.zeros(len(y), dtype=bool); held = train.copy()
        train[indices[a]], held[indices[b]] = True, True
        if np.any(train & held) or np.any((train | held) & test):
            raise AssertionError('Cross-fitting label leakage')
        result.append((train, held))
    if not np.array_equal(np.stack([h for _, h in result]).sum(0), fit.astype(int)):
        raise AssertionError('Every outer training row must receive exactly one held-out target')
    return result


def coalition_masks(modalities):
    return ((np.arange(2**modalities)[:, None] >> np.arange(modalities)) & 1).astype(np.float32)


def mean_marginal_gain(losses, modalities):
    """NLL(omit n) - NLL(include n), averaged over ALL other-modal subsets.

    losses shape [2**M, N]. Uniform coalitions give a Banzhaf-style average,
    NOT a Shapley value and NOT a biological causal effect.
    """
    if losses.shape[0] != 2**modalities or not np.isfinite(losses).all():
        raise ValueError('Invalid coalition losses')
    result = []
    ids = np.arange(2**modalities)
    for m in range(modalities):
        off = ids[(ids & (1 << m)) == 0]
        result.append((losses[off] - losses[off | (1 << m)]).mean(0))
    return np.stack(result, axis=1)


def training_class_weights(labels, fit, classes):
    counts = torch.bincount(labels[fit], minlength=classes).float()
    if bool((counts == 0).any()):
        raise ValueError('An inner training partition is missing a class')
    return 1 - counts/counts.sum()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def gain_supervision_loss(logits, gains, fit, temperature=.05):
    """Only outer TRAIN rows have finite cross-fitted gain targets."""
    targets = gains[fit].detach()
    if temperature <= 0 or not torch.isfinite(targets).all():
        raise ValueError('Missing/nonfinite train-only gain targets')
    return F.binary_cross_entropy_with_logits(logits[fit], (targets/temperature).sigmoid())
