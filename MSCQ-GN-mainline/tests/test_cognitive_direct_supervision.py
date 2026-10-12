"""Verify train-label isolation, path gradients and exact zero-weight control."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.cognitive_direct_supervision import classification_objective
from Model.cognitive_residual import CognitiveResidualModel


def rng_state():
    return (torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()])


class CognitiveDirectSupervisionTests(unittest.TestCase):
    def assert_rng_equal(self, left, right):
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertEqual(len(left[1]), len(right[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left[1], right[1])))

    def check_objective(self, device):
        torch.set_num_threads(1)
        final = torch.tensor([[1., -.1, .3], [.2, .5, -.3], [-.3, .6, .7],
                              [.8, -.2, -.9], [-.2, .1, .4]], device=device,
                             requires_grad=True)
        direct = (final.detach() * -.7 + .2).unsqueeze(1).clone().requires_grad_()
        output = dict(logits=final, unimodal_logits=direct)
        labels = torch.tensor([0, 1, 2, 1, 0], device=device)
        mask = torch.tensor([True, True, True, False, False], device=device)
        weights = torch.tensor([.88, .65, .47], device=device)
        before = rng_state()
        result = classification_objective(output, labels, mask, weights)
        self.assert_rng_equal(before, rng_state())
        self.assertEqual(set(result), {'total', 'main', 'direct'})
        main_expected = F.cross_entropy(final[mask], labels[mask], weight=weights,
                                        label_smoothing=.05)
        direct_expected = F.cross_entropy(direct[mask, 0], labels[mask],
                                          weight=weights, label_smoothing=.05)
        for key, expected in [('main', main_expected), ('direct', direct_expected),
                              ('total', main_expected + .05 * direct_expected)]:
            torch.testing.assert_close(result[key], expected, atol=0, rtol=0)

        # Invalid held-out labels make accidental full-cohort CE fail loudly.
        changed_labels = labels.clone()
        changed_labels[~mask] = -10000
        changed = classification_objective(output, changed_labels, mask, weights)
        for key in result:
            torch.testing.assert_close(result[key], changed[key], atol=0, rtol=0)
        changed_labels = labels.clone()
        changed_labels[0] = 2
        changed = classification_objective(output, changed_labels, mask, weights)
        self.assertGreater(float((changed['total'] - result['total']).abs()), 1e-5)

        final_grad, direct_grad = torch.autograd.grad(result['total'], (final, direct),
                                                      retain_graph=True)
        expected_final = torch.autograd.grad(main_expected, final, retain_graph=True)[0]
        expected_direct = torch.autograd.grad(direct_expected, direct, retain_graph=True)[0]
        torch.testing.assert_close(final_grad, expected_final, atol=0, rtol=0)
        torch.testing.assert_close(direct_grad, .05 * expected_direct, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(final_grad[~mask], torch.zeros_like(final_grad[~mask]),
                                   atol=0, rtol=0)
        torch.testing.assert_close(direct_grad[~mask], torch.zeros_like(direct_grad[~mask]),
                                   atol=0, rtol=0)
        isolated = torch.autograd.grad(result['direct'], (final, direct),
                                       allow_unused=True, retain_graph=True)
        self.assertIsNone(isolated[0])
        self.assertGreater(float(isolated[1].abs().sum()), 0)
        zero = classification_objective(output, labels, mask, weights, direct_weight=0)
        self.assertIs(zero['total'], zero['main'])
        zero_grad = torch.autograd.grad(zero['total'], (final, direct), allow_unused=True)
        self.assertIsNone(zero_grad[1])
        for invalid in (-.05, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                classification_objective(output, labels, mask, weights, invalid)

    def check_zero_weight_trajectory(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        x = x.to(device)
        original = CognitiveResidualModel(dd, cfg, companion='MRI', training_seed=0).to(device)
        candidate = deepcopy(original)
        labels = torch.arange(len(x), device=device) % dd['Class_Num']
        mask = torch.arange(len(x), device=device) < 5
        weights = torch.tensor([.88, .65, .47], device=device)
        optimizers = [torch.optim.Adam(model.parameters(), lr=.006, weight_decay=.0007)
                      for model in (original, candidate)]
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                       T_max=1000, eta_min=.0001) for optimizer in optimizers]
        for step in range(12):
            outcomes = []
            for index, (model, optimizer, scheduler) in enumerate(zip(
                    (original, candidate), optimizers, schedulers)):
                torch.manual_seed(31000 + step)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                if index == 0:
                    loss = F.cross_entropy(output['logits'][mask], labels[mask],
                                           weight=weights, label_smoothing=.05)
                else:
                    loss = classification_objective(output, labels, mask, weights,
                                                     direct_weight=0)['total']
                loss.backward()
                gradients = {name: p.grad.detach().clone()
                             for name, p in model.named_parameters()}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                               error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                outcomes.append((output['logits'].detach(), loss.detach(), gradients,
                                 rng_state()))
            torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=0, rtol=0)
            torch.testing.assert_close(outcomes[0][1], outcomes[1][1], atol=0, rtol=0)
            self.assert_rng_equal(outcomes[0][3], outcomes[1][3])
            for name in outcomes[0][2]:
                torch.testing.assert_close(outcomes[0][2][name], outcomes[1][2][name],
                                           atol=0, rtol=0, msg=name)
            for name, parameter in original.state_dict().items():
                torch.testing.assert_close(parameter, candidate.state_dict()[name],
                                           atol=0, rtol=0, msg=name)
        original.eval()
        candidate.eval()
        with torch.no_grad():
            torch.testing.assert_close(original(x)['logits'], candidate(x)['logits'],
                                       atol=0, rtol=0)

    def test_objective_cpu(self):
        self.check_objective('cpu')

    def test_zero_weight_trajectory_cpu(self):
        self.check_zero_weight_trajectory('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_objective_cuda(self):
        self.check_objective('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_zero_weight_trajectory_cuda(self):
        self.check_zero_weight_trajectory('cuda')


if __name__ == '__main__':
    unittest.main()
