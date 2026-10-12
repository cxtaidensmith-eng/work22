"""Check split isolation, block preservation and unchanged global RNG streams."""
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.modal_pairing import within_split_derangement, permute_modal_block


class PairingTests(unittest.TestCase):
    def test_mapping_and_random_streams(self):
        train = np.arange(17) < 12
        before = (pickle.dumps(np.random.get_state()), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        mapping = within_split_derangement(train, ~train, 71101)
        np.testing.assert_array_equal(mapping, within_split_derangement(train, ~train, 71101))
        self.assertFalse(np.any(mapping == np.arange(17)))
        np.testing.assert_array_equal(train[mapping], train)
        np.testing.assert_array_equal(np.sort(mapping), np.arange(17))
        self.assertEqual(before[0], pickle.dumps(np.random.get_state()))
        self.assertTrue(torch.equal(before[1], torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[2],torch.cuda.get_rng_state_all())))

    def check_device(self, device):
        x = torch.arange(17*9,dtype=torch.float32,device=device).reshape(17,9)
        original = x.clone()
        train = np.arange(17) < 12
        mapping = within_split_derangement(train,~train,71101)
        cols = [1,3,8]
        changed = permute_modal_block(x,cols,mapping)
        torch.testing.assert_close(x,original,atol=0,rtol=0)
        others = [j for j in range(9) if j not in cols]
        torch.testing.assert_close(changed[:,others],x[:,others],atol=0,rtol=0)
        torch.testing.assert_close(changed[:,cols],x[torch.tensor(mapping,device=device)][:,cols],atol=0,rtol=0)
        # Whole rows, not independent feature-column shuffles, within each split.
        for mask in (train,~train):
            indices = np.flatnonzero(mask)
            donors = mapping[indices]
            torch.testing.assert_close(changed[torch.tensor(indices,device=device)][:,cols],
                                      x[torch.tensor(donors,device=device)][:,cols],atol=0,rtol=0)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__=='__main__':
    unittest.main()
