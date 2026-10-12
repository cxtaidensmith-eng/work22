"""Label-free, fixed within-split donor mappings for modal pairing controls."""
import numpy as np
import torch


def within_split_derangement(train_mask, test_mask, mapping_seed):
    train = np.asarray(train_mask, dtype=bool)
    test = np.asarray(test_mask, dtype=bool)
    if train.ndim != 1 or train.shape != test.shape or np.any(train & test) or not np.all(train | test):
        raise ValueError('Train and test masks must partition the rows')
    mapping = np.arange(len(train), dtype=np.int64)
    for role, mask in enumerate((train, test)):
        rows = np.flatnonzero(mask)
        if len(rows) < 2:
            raise ValueError('Each split needs at least two rows')
        rng = np.random.default_rng(np.random.SeedSequence([int(mapping_seed), role]))
        for _ in range(10000):
            donor = rng.permutation(rows)
            if np.all(donor != rows):
                mapping[rows] = donor
                break
        else:
            raise RuntimeError('Could not produce a derangement')
    assert np.array_equal(np.sort(mapping), np.arange(len(train)))
    assert not np.any(mapping == np.arange(len(train)))
    assert np.array_equal(train[mapping], train)
    return mapping


def permute_modal_block(features, columns, donor_rows):
    """Copy only one modality from its donor; original feature tensor unchanged."""
    if features.ndim != 2:
        raise ValueError('Expected a matrix')
    rows = torch.as_tensor(donor_rows, dtype=torch.long, device=features.device)
    cols = torch.as_tensor(columns, dtype=torch.long, device=features.device)
    if rows.shape != (len(features),) or len(cols) == 0:
        raise ValueError('Invalid donor mapping or modality columns')
    if not torch.equal(torch.sort(rows).values, torch.arange(len(features), device=features.device)):
        raise ValueError('Donor mapping must be a permutation')
    result = features.clone()
    result[:, cols] = features.index_select(0, rows).index_select(1, cols)
    return result
