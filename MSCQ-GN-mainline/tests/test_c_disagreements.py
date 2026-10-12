import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case
from Model.reference_selection import ReferenceSelectionModel
from diagnose_c_disagreements import FrozenFlow, norm_match, overridden_logits


class FixedInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.x, graphs, dd, cfg = tiny_case()
        self.model = ReferenceSelectionModel(dd, cfg, graphs, learned=False).eval()
        self.flow = FrozenFlow(self.model, self.x)

    @torch.no_grad()
    def test_identity_and_closed_recover_existing_inference_paths(self):
        flow = self.flow
        same = overridden_logits(self.model, self.x, flow.replace_foreign(flow.foreign_values(flow.values)))
        torch.testing.assert_close(same, self.model(self.x)[0], atol=1e-6, rtol=1e-6)
        closed = overridden_logits(self.model, self.x, flow.replace_foreign(torch.zeros_like(flow.neighbor)))
        torch.testing.assert_close(closed, self.model(self.x, torch.zeros(1, flow.m))[0], atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_foreign_replacement_excludes_recipients_own_value(self):
        flow = self.flow
        original = flow.foreign_values(flow.values)
        modified = flow.values.clone().reshape(flow.m, flow.m, flow.count, 2, flow.h//2)
        modified[:, :, 0] += 10
        changed = flow.foreign_values(modified.reshape_as(flow.values))
        # Recipient zero's value only belongs to the removed diagonal block.
        torch.testing.assert_close(original[0], changed[0], atol=2e-6, rtol=2e-6)
        self.assertGreater(float((original[1:]-changed[1:]).norm()), .01)

    def test_norm_control_preserves_magnitude_and_reports_zero_directions(self):
        reference = torch.randn(4, 2, 8)
        replacement = torch.randn_like(reference)
        replacement[0, 0] = 0
        matched, invalid = norm_match(reference, replacement)
        self.assertEqual(int(invalid.sum()), 1)
        self.assertTrue(torch.isfinite(matched).all())
        torch.testing.assert_close(matched.norm(dim=-1)[~invalid], reference.norm(dim=-1)[~invalid])
        torch.testing.assert_close(matched[0, 0], torch.zeros(8))


if __name__ == '__main__':
    unittest.main()
