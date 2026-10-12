"""Check independence, matched stochastic inputs and trainability of controls."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.modal_evidence import ModalEvidenceModel
from Model.single_modal_evidence import SingleModalEvidenceModel


class SingleModalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x, self.dd, self.cfg = tiny_case()

    def check_device(self, device):
        x = self.x.to(device)
        parent = ModalEvidenceModel(self.dd, self.cfg, 'additive', 2).to(device)
        for m, name in enumerate(self.dd['Modal_Name']):
            before = torch.get_rng_state().clone()
            model = SingleModalEvidenceModel(self.dd, self.cfg, name, 2).to(device)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for train in (False, True):
                parent.train(train)
                model.train(train)
                torch.manual_seed(918)
                expected = parent(x)['unimodal_logits'][:, m]
                state = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
                torch.manual_seed(918)
                actual = model(x)['logits']
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                self.assertTrue(torch.equal(state[0], torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(state[1], torch.cuda.get_rng_state_all())))
                changed = x.clone()
                others = [i for i in range(x.shape[1]) if i not in self.dd['Modal_Index'][m]]
                changed[:, others] = 1234.
                torch.manual_seed(918)
                torch.testing.assert_close(actual, model(changed)['logits'], atol=0, rtol=0)
            model.eval()
            probe = x.clone().requires_grad_(True)
            model(probe)['logits'].square().sum().backward()
            self.assertEqual(int(torch.count_nonzero(probe.grad[:, others])), 0)
            self.assertGreater(float(probe.grad[:, self.dd['Modal_Index'][m]].abs().sum()), 0)
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            old = model(x)['logits'].detach()
            changed = x.clone()
            changed[0] += 9
            torch.testing.assert_close(old[1:], model(changed)['logits'][1:], atol=0, rtol=0)
            self.assertEqual(model.parameter_counts()['total'], sum(p.numel() for p in model.parameters()))

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
