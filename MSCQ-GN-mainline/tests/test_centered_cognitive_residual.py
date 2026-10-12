"""Verify token centering, unchanged randomness and direct supervision."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.centered_cognitive_residual import CenteredCognitiveResidualModel
from Model.cognitive_direct_supervision import classification_objective
from Model.cognitive_residual import CognitiveResidualModel


def rng_state():
    return (torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()])


class CenteredCognitiveResidualTests(unittest.TestCase):
    def assert_rng_equal(self, left, right):
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertEqual(len(left[1]), len(right[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left[1], right[1])))

    def check_device(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        dd['Modal_Name'][0] = 'PET'
        x = x.to(device)
        original = CognitiveResidualModel(dd, cfg).to(device)
        before = rng_state()
        model = CenteredCognitiveResidualModel(dd, cfg).to(device)
        self.assert_rng_equal(before, rng_state())
        self.assertEqual(original.parameter_counts(), model.parameter_counts())
        self.assertEqual(original.state_dict().keys(), model.state_dict().keys())
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name],
                                       atol=0, rtol=0, msg=name)

        # The initial supplement remains zero, including training dropout.
        for training in (False, True):
            original.train(training); model.train(training)
            torch.manual_seed(2300)
            expected = original(x)
            after = rng_state()
            torch.manual_seed(2300)
            actual = model(x)
            self.assert_rng_equal(after, rng_state())
            for key in ('logits', 'tokens', 'unimodal_logits', 'pair_logits'):
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)

        # Exercise a nonzero supplement rather than testing only zero init.
        with torch.no_grad():
            values = torch.linspace(-.5, .7, model.head[-1].weight.numel(),
                                    device=device).reshape_as(model.head[-1].weight)
            model.head[-1].weight.copy_(values)
            model.head[-1].bias.copy_(torch.tensor([.2, -.3, .4], device=device))
        original.load_state_dict(model.state_dict(), strict=True)
        for training in (False, True):
            original.train(training); model.train(training)
            torch.manual_seed(2301)
            expected = original(x)
            after = rng_state()
            torch.manual_seed(2301)
            calls = []
            hook = model.head.register_forward_hook(lambda *args: calls.append(1))
            actual = model(x)
            hook.remove()
            self.assertEqual(len(calls), 2, 'Only observed and reference head evaluations')
            self.assert_rng_equal(after, rng_state())
            torch.testing.assert_close(expected['tokens'], actual['tokens'], atol=0, rtol=0)
            torch.testing.assert_close(expected['unimodal_logits'], actual['unimodal_logits'], atol=0, rtol=0)
            torch.testing.assert_close(expected['pair_logits'][:, 0], actual['raw_joint_logits'], atol=0, rtol=0)
            reference_tokens = actual['tokens'].clone()
            reference_tokens[:, model.modalities.index('PET')] = 0
            reference = model.head(reference_tokens.flatten(start_dim=1))
            torch.testing.assert_close(reference, actual['reference_joint_logits'], atol=0, rtol=0)
            centered = actual['raw_joint_logits'] - reference
            torch.testing.assert_close(centered, actual['pair_logits'][:, 0], atol=0, rtol=0)
            torch.testing.assert_close(actual['logits'], actual['unimodal_logits'][:, 0] + centered, atol=0, rtol=0)

        model.eval()
        original_output = model(x)
        changed = x.clone()
        changed[:, dd['Modal_Index'][0]] += torch.tensor([3., -2., 5.], device=device)
        changed_output = model(changed)
        torch.testing.assert_close(original_output['unimodal_logits'], changed_output['unimodal_logits'], atol=0, rtol=0)
        self.assertGreater(float((original_output['logits'] - changed_output['logits']).abs().max()), 1e-7)

        # At a zero companion token, values and cognitive-only derivatives
        # cancel. The derivative with respect to a hypothetical nonzero PET
        # token need not vanish and is deliberately not constrained here.
        tokens = original_output['tokens'].detach().clone()
        tokens[:, model.modalities.index('PET')] = 0
        tokens.requires_grad_(True)
        supplement = model.supplement_from_tokens(tokens)
        torch.testing.assert_close(supplement, torch.zeros_like(supplement), atol=0, rtol=0)
        parameters = tuple(model.head.parameters())
        gradients = torch.autograd.grad(supplement.square().sum() + supplement.sum(),
                                        (tokens,) + parameters)
        cognitive_gradient = gradients[0][:, model.cognitive_index]
        torch.testing.assert_close(cognitive_gradient, torch.zeros_like(cognitive_gradient), atol=0, rtol=0)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)

        # Check that cancellation also reaches the upstream cognitive encoder.
        raw = x.detach().clone().requires_grad_(True)
        tokens = model(raw)['tokens'].clone()
        tokens[:, model.modalities.index('PET')] = 0
        supplement = model.supplement_from_tokens(tokens)
        cognitive_parameters = tuple(model.encoders[model.cognitive_index].parameters())
        gradients = torch.autograd.grad(supplement.sum(), (raw,) + cognitive_parameters)
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)

        labels = torch.arange(len(x), device=device) % dd['Class_Num']
        mask = torch.arange(len(x), device=device) < 5
        weights = torch.tensor([.88, .65, .47], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.006, weight_decay=.0007)
        for step in range(3):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            losses = classification_objective(output, labels, mask, weights)
            main = F.cross_entropy(output['logits'][mask], labels[mask], weight=weights, label_smoothing=.05)
            direct = F.cross_entropy(output['unimodal_logits'][mask, 0], labels[mask], weight=weights, label_smoothing=.05)
            torch.testing.assert_close(losses['total'], main + .05 * direct, atol=0, rtol=0)
            losses['total'].backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            # The joint head's final bias cancels for every token, not just 0.
            torch.testing.assert_close(model.head[-1].bias.grad,
                                       torch.zeros_like(model.head[-1].bias), atol=0, rtol=0)
            self.assertGreater(float(model.head[0].weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.cognitive_head[0].weight.grad.abs().sum()), 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
