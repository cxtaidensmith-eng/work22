"""Input isolation, matched stochastic tokens, and gradient checks."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.modal_evidence import ModalEvidenceModel
from Model.subset_fusion import SubsetFusionModel


class SubsetFusionTests(unittest.TestCase):
    def check_device(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        x = x.to(device)
        full = ModalEvidenceModel(dd, cfg, 'fusion', 0).to(device)
        for chosen in [('COGNITIVE_TEST',), ('COGNITIVE_TEST', 'MRI'),
                       ('COGNITIVE_TEST', 'ROI_AVERAGE'), tuple(dd['Modal_Name'])]:
            before = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            model = SubsetFusionModel(dd, cfg, chosen, 0).to(device)
            self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before[1], torch.cuda.get_rng_state_all())))
            selected = [dd['Modal_Name'].index(name) for name in model.modalities]
            features = [i for m in selected for i in dd['Modal_Index'][m]]
            unused = [i for i in range(x.shape[1]) if i not in features]
            self.assertEqual(model.fusion_width, full.fusion_width)
            for training in (False, True):
                full.train(training); model.train(training)
                torch.manual_seed(991)
                original = full(x)
                after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
                torch.manual_seed(991)
                actual = model(x)
                torch.testing.assert_close(actual['tokens'], original['tokens'][:, selected], atol=2e-6, rtol=2e-6)
                masked = torch.zeros_like(original['tokens'])
                masked[:, selected] = original['tokens'][:, selected]
                torch.testing.assert_close(actual['logits'], full.fusion_head(masked.flatten(1)), atol=2e-6, rtol=2e-6)
                self.assertTrue(torch.equal(after[0], torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(after[1], torch.cuda.get_rng_state_all())))
                changed = x.clone(); changed[:, unused] = 1234
                torch.manual_seed(991)
                torch.testing.assert_close(actual['logits'], model(changed)['logits'], atol=0, rtol=0)
            model.eval()
            probe = x.clone().requires_grad_(True)
            model(probe)['logits'].square().sum().backward()
            self.assertEqual(int(torch.count_nonzero(probe.grad[:, unused])), 0)
            for feature_group in selected:
                self.assertGreater(float(probe.grad[:, dd['Modal_Index'][feature_group]].abs().sum()), 0)
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            old = model(x)['logits'].detach()
            changed = x.clone(); changed[0] += 9
            torch.testing.assert_close(old[1:], model(changed)['logits'][1:], atol=0, rtol=0)
            self.assertEqual(model.parameter_counts()['total'], sum(p.numel() for p in model.parameters()))

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
