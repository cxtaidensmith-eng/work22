import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel, RelationContentPropagation, modal_candidates
from Model.selection_only import SelectionOnlyModel, SelectionOnlyPropagation


class SelectionOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_fixed_matches_original_values_and_all_gradients_after_weight_changes(self):
        x = torch.randn(7, 6)
        indices = [[0, 1, 2], [3, 4, 5]]
        graphs = modal_candidates(x, indices, k=2)
        original = RelationContentPropagation(8, 2, 7, graphs, 'content').double()
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.add_(torch.randn_like(parameter) * .2)
        fixed = SelectionOnlyPropagation(8, 2, 7, graphs, x, indices, 'fixed', k=2).double()
        fixed.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(set(dict(original.named_parameters())), set(dict(fixed.named_parameters())))
        z = torch.randn(7, 2, 8, dtype=torch.double, requires_grad=True)
        probe = torch.randn_like(z)
        a, b = original(z), fixed(z)
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        ga = torch.autograd.grad((a*probe).sum(), (z, *original.parameters()))
        gb = torch.autograd.grad((b*probe).sum(), (z, *fixed.parameters()))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)

    def test_beta_changes_selection_but_not_weights_or_forced_fixed_predictions(self):
        x = torch.randn(7, 6)
        indices = [[0, 1, 2], [3, 4, 5]]
        graphs = modal_candidates(x, indices, k=2)
        layer = SelectionOnlyPropagation(8, 2, 7, graphs, x, indices, 'learned', k=2).eval()
        z = torch.randn(7, 2, 8)
        affinity = torch.randn(2, 7, 7)
        before_weight = layer.edge_weights(affinity).detach().clone()
        layer(z)
        original_support = layer.last_support.clone()
        with torch.no_grad():
            layer.relation_blend.fill_(-5)
        layer(z)
        self.assertTrue(bool((original_support ^ layer.last_support).any()))
        torch.testing.assert_close(before_weight, layer.edge_weights(affinity), rtol=0, atol=0)
        layer.topology_mode = 'fixed'
        first = layer(z)
        with torch.no_grad():
            layer.relation_blend.fill_(5)
        torch.testing.assert_close(first, layer(z), rtol=0, atol=0)
        grad = torch.autograd.grad(layer(z).square().sum(), layer.relation_blend, allow_unused=True)[0]
        self.assertIsNone(grad)
        layer.topology_mode = 'learned'
        layer.train()
        grad = torch.autograd.grad((layer(z)*torch.randn_like(z)).sum(), layer.relation_blend)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_initial_models_rng_active_parameters_and_training(self):
        x, graphs, dd, cfg = tiny_case()
        models, states = [], []
        for mode in ('fixed', 'learned', 'original'):
            torch.manual_seed(9)
            model = (RelationSheafModel(dd, cfg, graphs, 'content', training_seed=9) if mode == 'original'
                else SelectionOnlyModel(dd, cfg, graphs, x, 9, mode, k=2))
            models.append(model.eval())
            states.append(torch.get_rng_state())
        self.assertTrue(all(torch.equal(states[0], s) for s in states))
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),
                         sum(p.numel() for p in models[2].parameters()))
        self.assertEqual(sum(p.numel() for p in models[1].parameters()),
                         sum(p.numel() for p in models[0].parameters()) + 2)
        for model in models[:2]:
            for name, value in models[2].state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(model(x)[0], models[2](x)[0], atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(model.propagation.last_support, model.propagation.fixed_support))
        torch.testing.assert_close(models[0](x)[0], models[1](x)[0], rtol=0, atol=0)
        for model in models[:2]:
            model.train()
            logits, embeddings, auxiliary = model(x)
            (F.cross_entropy(logits, torch.arange(7) % 3) +
             sum(t.square().mean() for t in [*embeddings, *auxiliary])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(models[1].propagation.relation_blend.grad.abs().sum()), 0.)

    def test_checkpoint_roundtrip_and_full_cohort(self):
        x, graphs, dd, cfg = tiny_case()
        for mode in ('fixed', 'learned'):
            first = SelectionOnlyModel(dd, cfg, graphs, x, 0, mode, k=2).eval()
            if mode == 'learned':
                with torch.no_grad(): first.propagation.relation_blend.fill_(.7)
            second = SelectionOnlyModel(dd, cfg, graphs, x, 0, mode, k=2).eval()
            second.load_state_dict(first.state_dict())
            torch.testing.assert_close(first(x)[0], second(x)[0], atol=0, rtol=0)
            self.assertTrue(torch.equal(first.propagation.last_neighbors, second.propagation.last_neighbors))
            with self.assertRaises(ValueError): first(x[:-1])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA RNG equivalence requires CUDA')
    def test_cuda_rng_matches_original_before_and_after_training_forward(self):
        x, graphs, dd, cfg = tiny_case()
        states=[];after=[]
        for mode in ('original', 'fixed', 'learned'):
            torch.manual_seed(9)
            model=(RelationSheafModel(dd,cfg,graphs,'content',training_seed=9) if mode=='original'
                else SelectionOnlyModel(dd,cfg,graphs,x,9,mode,k=2)).cuda().train()
            states.append(torch.cuda.get_rng_state_all())
            model(x.cuda())
            after.append(torch.cuda.get_rng_state_all())
        for results in (states,after):
            for other in results[1:]:
                self.assertEqual(len(results[0]),len(other))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(results[0],other)))


if __name__ == '__main__': unittest.main()
