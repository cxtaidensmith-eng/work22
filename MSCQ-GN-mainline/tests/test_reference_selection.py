import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case, dense_operator
from Model.relation_sheaf import RelationSheafModel, conformal_maps, modal_candidates
from Model.reference_selection import composed_self_part, ReferenceSelectionModel
from Utils.reference_selection import (rewire_candidates, inner_splits, coalition_masks,
    mean_marginal_gain, training_class_weights, gain_supervision_loss)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_composed_diagonal_and_gradients_match_dense_operator(self):
        edge = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        x = torch.randn(3, 2, 4, dtype=torch.double, requires_grad=True)
        p = (torch.randn(3, 4, dtype=torch.double)*.2).requires_grad_()
        v = torch.rand(3, dtype=torch.double, requires_grad=True)
        w = v[torch.tensor([0, 1, 0, 2, 1, 2])]
        actual = composed_self_part(x, edge, w, *conformal_maps(p))
        expected = torch.stack([dense_operator(x * torch.eye(3, dtype=x.dtype)[i, :, None, None],
                                 edge, w, p)[i] for i in range(3)])
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        probe = torch.randn_like(x)
        ga = torch.autograd.grad((probe*actual).sum(), (x, p, v), retain_graph=True)
        gb = torch.autograd.grad((probe*expected).sum(), (x, p, v))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_self_component_has_no_foreign_input_for_fixed_operator(self):
        edge = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        x = torch.randn(3, 2, 4)
        maps = conformal_maps(torch.randn(3, 4)*.2)
        a = composed_self_part(x, edge, torch.ones(6), *maps)
        x[1:] += 100
        b = composed_self_part(x, edge, torch.ones(6), *maps)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)

    def test_constant_map_factorization_preserves_values_and_parameter_gradients(self):
        edge = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        x = torch.randn(3, 2, 4, dtype=torch.double, requires_grad=True)
        p = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
        w = torch.rand(3, dtype=torch.double, requires_grad=True)
        weights = w[torch.tensor([0, 1, 0, 2, 1, 2])]
        maps = conformal_maps(p.expand(3, -1))
        full = composed_self_part(x, edge, weights, *maps)
        fast = composed_self_part(x, edge, weights, *maps, constant_maps=True)
        torch.testing.assert_close(full, fast, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(x)
        full_grad = torch.autograd.grad((full * probe).sum(), (x, p, w), retain_graph=True)
        fast_grad = torch.autograd.grad((fast * probe).sum(), (x, p, w))
        for a, b in zip(full_grad, fast_grad):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_open_gate_matches_C_weights_rng_outputs_and_gradients(self):
        x, graphs, dd, cfg = tiny_case()
        torch.manual_seed(0)
        c = RelationSheafModel(dd, cfg, graphs, 'content', training_seed=0).eval()
        state = torch.get_rng_state()
        torch.manual_seed(0)
        gated = ReferenceSelectionModel(dd, cfg, graphs, training_seed=0).eval()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in c.state_dict().items():
            torch.testing.assert_close(value, gated.state_dict()[key], rtol=0, atol=0)
        expected = c(x)
        actual = gated(x, reference_mask=torch.ones(1, 2))
        for a, b in zip([expected[0], *expected[1], *expected[2]], [actual[0], *actual[1], *actual[2]]):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        expected[0].square().sum().backward()
        actual[0].square().sum().backward()
        gp = dict(gated.named_parameters())
        for name, p in c.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, gp[name].grad, rtol=1e-5, atol=1e-6)

    def test_learned_gate_backprop_and_exact_closed_endpoint(self):
        x, graphs, dd, cfg = tiny_case()
        model = ReferenceSelectionModel(dd, cfg, graphs).eval()
        before = model(x)[0]
        before.square().sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.propagation.selector.parameters()), 0)
        with torch.no_grad():
            model.propagation.selector[-1].weight.zero_()
            model.propagation.selector[-1].bias.fill_(-50)
            closed = model(x)[0]
            self.assertEqual(float(model.propagation.last_gate.sum()), 0)
            forced = model(x, torch.zeros(1, 2))[0]
            torch.testing.assert_close(closed, forced, rtol=0, atol=0)

    def test_random_graph_preserves_each_degree_and_endpoint_cosine(self):
        x = torch.randn(40, 8)
        idx = [list(range(4)), list(range(4, 8))]
        gs = modal_candidates(x, idx, 4)
        first, stats = rewire_candidates(gs, x, idx, 42)
        second, _ = rewire_candidates(gs, x, idx, 42)
        for g, a, b, dims in zip(gs, first, second, idx):
            self.assertTrue(torch.equal(g['degree'], a['degree']))
            self.assertTrue(torch.equal(a['edge'], b['edge']))
            self.assertFalse(torch.equal(a['edge'], g['edge']))
            pairs = set(map(tuple, a['edge'].T.tolist()))
            self.assertTrue(all(i != j and (j, i) in pairs for i, j in pairs))
            z = torch.nn.functional.normalize(x[:, dims], dim=-1)
            torch.testing.assert_close(a['similarity'], (z @ z.T)[a['edge'][0], a['edge'][1]])

    def test_crossfit_targets_never_use_outer_test_labels(self):
        y = torch.arange(60) % 3
        fit = torch.arange(60) < 48
        splits = inner_splits(y, fit, ~fit, 51)
        changed = y.clone(); changed[~fit] = (changed[~fit] + 1) % 3
        again = inner_splits(changed, fit, ~fit, 51)
        for (a, b), (c, d) in zip(splits, again):
            np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
            tensor_a = torch.from_numpy(a)
            labels = y.clone(); labels[~tensor_a] = (labels[~tensor_a]+1) % 3
            torch.testing.assert_close(training_class_weights(y, tensor_a, 3),
                                       training_class_weights(labels, tensor_a, 3))

    def test_marginal_gain_sign_and_interaction(self):
        masks = coalition_masks(2)
        # Modal 0 reduces loss by .3, modal 1 by .2, joint interaction by .4.
        losses = (2 - .3*masks[:, 0] - .2*masks[:, 1] - .4*masks.prod(1))[:, None]
        np.testing.assert_allclose(mean_marginal_gain(losses, 2), [[.5, .4]], atol=1e-6)

    def test_auxiliary_gain_loss_masks_test_rows_and_gradient(self):
        logits = torch.zeros(8, 2, requires_grad=True)
        fit = torch.arange(8) < 6
        targets = torch.full((8, 2), float('nan'))
        targets[fit] = torch.tensor([.1, -.1])
        loss = gain_supervision_loss(logits, targets, fit)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[~fit], torch.zeros_like(logits.grad[~fit])))
        self.assertLess(float(logits.grad[fit, 0].sum()), 0)
        self.assertGreater(float(logits.grad[fit, 1].sum()), 0)


if __name__ == '__main__':
    unittest.main()
