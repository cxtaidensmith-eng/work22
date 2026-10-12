"""Operator, gradient, cohort alignment, and shared-initialization checks."""
import sys
from pathlib import Path
import unittest
import tempfile
import torch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.relation_sheaf import (modal_candidates, conformal_maps, cooperative_operator,
                                  RelationContentPropagation, RelationSheafModel)
from Utils.graph_diagnostics import GraphDiagnostics


def tiny_case():
    x = torch.arange(42).float().reshape(7, 6)/42
    graphs = modal_candidates(x, [[0, 1, 2], [3, 4, 5]], k=2)
    dd = dict(Modal_Name=['MRI', 'COGNITIVE_TEST'], Modal_Index=[[0, 1, 2], [3, 4, 5]],
              Class_Num=3, Feature_Num=6, In_Channels=6, Sample_Num=7, Adj=torch.eye(7))
    cfg = SimpleNamespace(Hidden_size=8, Drop_rate=.2, ChebGCN_K=3, num_layers=1,
        num_heads=2, input_noise_std=.1, drop_path=.1, global_word_emb=8,
        semantic_branch='both', adj_mode='learned', modal_structure='modal',
        readout_mode='class_query', seed=0)
    return x, graphs, dd, cfg


def dense_operator(x, edge, weight, parameters):
    n = len(x)
    send, receive, cs, ct = conformal_maps(parameters)
    adjacency = x.new_zeros(n, n).index_put(tuple(edge), weight)
    degree = adjacency.sum(1)
    blocks = []
    for i in range(n):
        blocks.append(torch.cat([-adjacency[i, j] * receive[i].T @ send[j]
                                 for j in range(n)], dim=1))
    off = torch.cat(blocks, dim=0)
    ops = []
    for scale in (cs, ct):
        d = (degree * scale.square()).repeat_interleave(2)
        inv = (d + 1).rsqrt()
        ops.append(inv[:, None] * (off + torch.diag(d)) * inv[None, :])
    return (ops[1] @ ops[0] @ x.reshape(n*2, -1)).reshape_as(x)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)
        self.edge = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])

    def test_sparse_composition_and_gradients_against_dense_blocks(self):
        x = torch.randn(3, 2, 4, dtype=torch.double, requires_grad=True)
        p = (torch.randn(3, 4, dtype=torch.double)*.2).requires_grad_()
        v = torch.rand(3, dtype=torch.double, requires_grad=True)
        w = v[torch.tensor([0, 1, 0, 2, 1, 2])]
        actual = cooperative_operator(x, self.edge, w, *conformal_maps(p))
        expected = dense_operator(x, self.edge, w, p)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        probe = torch.randn_like(actual)
        grads_a = torch.autograd.grad((actual*probe).sum(), (x, p, v), retain_graph=True)
        grads_b = torch.autograd.grad((expected*probe).sum(), (x, p, v))
        for a, b in zip(grads_a, grads_b):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_identity_maps_reduce_to_normalized_laplacian_squared(self):
        x = torch.randn(3, 2, 4, dtype=torch.double)
        p = torch.zeros(3, 4, dtype=torch.double)
        w = torch.ones(6, dtype=torch.double)
        adjacency = torch.ones(3, 3, dtype=torch.double) - torch.eye(3, dtype=torch.double)
        lap = (2*torch.eye(3, dtype=torch.double) - adjacency)/3
        expected = (lap @ lap @ x.flatten(1)).reshape_as(x)
        torch.testing.assert_close(cooperative_operator(x, self.edge, w, *conformal_maps(p)), expected)

    def test_zero_receiver_prevents_update_for_fixed_maps(self):
        x = torch.randn(3, 2, 4)
        p = torch.randn(3, 4)*.2
        send, receive, cs, ct = conformal_maps(p)
        receive[0] = 0
        ct[0] = 0
        result = cooperative_operator(x, self.edge, torch.ones(6), send, receive, cs, ct)
        torch.testing.assert_close(result[0], torch.zeros_like(result[0]))

    def test_candidates_exclude_self_and_are_symmetric_and_stable(self):
        x = torch.ones(7, 6)  # all similarities tied: deterministic row-order policy
        a = modal_candidates(x, [[0, 1, 2], [3, 4, 5]], k=2)
        b = modal_candidates(x, [[0, 1, 2], [3, 4, 5]], k=2)
        for ga, gb in zip(a, b):
            self.assertTrue(torch.equal(ga['edge'], gb['edge']))
            edges = set(map(tuple, ga['edge'].T.tolist()))
            self.assertTrue(all(i != j and (j, i) in edges for i, j in edges))
            self.assertEqual(ga['neighbors'][0].tolist(), [1, 2])
            self.assertTrue(bool((ga['degree'] >= 2).all()))

    def test_static_maps_ignore_patient_state_dynamic_maps_use_it(self):
        x = torch.randn(7, 6)
        graphs = modal_candidates(x, [[0, 1, 2], [3, 4, 5]], k=2)
        z = torch.randn(7, 2, 8)
        for mode in ('shared', 'content', 'static', 'dynamic'):
            layer = RelationContentPropagation(8, 2, 7, graphs, mode)
            first = layer.map_values(z)
            second = layer.map_values(z + torch.randn_like(z))
            self.assertEqual(torch.equal(first, second), mode != 'dynamic')
            result = layer(z)
            result.square().sum().backward()
            self.assertTrue(torch.isfinite(result).all())
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_content_control_uses_equal_active_budget_and_ignores_relation(self):
        graphs = modal_candidates(torch.randn(7, 6), [[0, 1, 2], [3, 4, 5]], k=2)
        content = RelationContentPropagation(8, 2, 7, graphs, 'content')
        static = RelationContentPropagation(8, 2, 7, graphs, 'static')
        self.assertEqual(sum(p.numel() for p in content.parameters()),
                         sum(p.numel() for p in static.parameters()))
        z = torch.randn(7, 2, 8)
        maps = content.map_values(z)
        torch.testing.assert_close(maps[0], maps[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(maps[:, 0], maps[:, 1]))
        static_maps = static.map_values(z)
        self.assertFalse(torch.equal(static_maps[0], static_maps[1]))
        (maps * torch.randn_like(maps)).sum().backward()
        for parameter in list(content.content_id.parameters()) + list(content.conditioner.parameters()):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_common_weights_and_rng_identical_and_checkpoint_roundtrip(self):
        x = torch.randn(7, 6)
        graphs = modal_candidates(x, [[0, 1, 2], [3, 4, 5]], k=2)
        dd = dict(Modal_Name=['MRI', 'COGNITIVE_TEST'], Modal_Index=[[0, 1, 2], [3, 4, 5]],
                  Class_Num=3, Feature_Num=6, In_Channels=6, Sample_Num=7, Adj=torch.eye(7))
        cfg = SimpleNamespace(Hidden_size=8, Drop_rate=.2, ChebGCN_K=3, num_layers=1,
            num_heads=2, input_noise_std=.1, drop_path=.1, global_word_emb=8,
            semantic_branch='both', adj_mode='learned', modal_structure='modal',
            readout_mode='class_query', seed=0)
        models, rng = [], []
        for mode in ('shared', 'content', 'static', 'dynamic'):
            torch.manual_seed(0)
            models.append(RelationSheafModel(dd, cfg, graphs, mode).eval())
            rng.append(torch.get_rng_state())
        for other, state in zip(models[1:], rng[1:]):
            self.assertTrue(torch.equal(rng[0], state))
            other_params = dict(other.named_parameters())
            for key, value in models[0].named_parameters():
                if key in other_params:
                    torch.testing.assert_close(value, other_params[key], rtol=0, atol=0)
        for mode, model in zip(('shared', 'content', 'static', 'dynamic'), models):
            clone = RelationSheafModel(dd, cfg, graphs, mode).eval()
            clone.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(x)[0], clone(x)[0], rtol=0, atol=0)
            model.train()
            outputs = model(x)
            sum(t.square().mean() for t in [outputs[0], *outputs[1], *outputs[2]]).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

        # Repeat seed changes ALL trained components without changing cfg.seed,
        # which exclusively controls data order and the original folds.
        repeated = []
        for mode in ('shared', 'content', 'static'):
            torch.manual_seed(1)
            repeated.append(RelationSheafModel(dd, cfg, graphs, mode, training_seed=1))
        for other in repeated[1:]:
            base_params = dict(repeated[0].named_parameters())
            for key, value in other.named_parameters():
                if key in base_params:
                    torch.testing.assert_close(value, base_params[key], rtol=0, atol=0)
        for key in ('classifier.0.weight', 'propagation.value.weight', 'modal_token_encoder.modal_proj.0.0.weight'):
            self.assertFalse(torch.equal(dict(models[0].named_parameters())[key],
                                         dict(repeated[0].named_parameters())[key]))
        self.assertEqual(cfg.seed, 0)

    def test_self_only_preserves_common_weights_and_has_no_cross_patient_messages(self):
        x, graphs, dd, cfg = tiny_case()
        torch.manual_seed(0)
        graph = RelationSheafModel(dd, cfg, graphs, 'static').eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(0)
        own = RelationSheafModel(dd, cfg, graphs, 'self_only').eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertLess(sum(p.numel() for p in own.parameters()), sum(p.numel() for p in graph.parameters()))
        graph_state = graph.state_dict()
        for key, value in own.state_dict().items():
            torch.testing.assert_close(value, graph_state[key], rtol=0, atol=0)
        self.assertEqual(list(own.propagation.state_dict()), ['output_norm.weight', 'output_norm.bias'])
        with torch.no_grad():
            graph.propagation.step_logit.fill_(-100)
            torch.testing.assert_close(own(x)[0], graph(x)[0], rtol=0, atol=0)
            changed = x.clone()
            changed[1:] += 50
            torch.testing.assert_close(own(x)[0][0], own(changed)[0][0], rtol=0, atol=0)
        own.train()
        result = own(x)
        sum(t.square().mean() for t in [result[0], *result[1], *result[2]]).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in own.parameters()))

    def test_passive_observer_preserves_updates_and_training_rng(self):
        x, graphs, dd, cfg = tiny_case()
        models, optimizers = [], []
        for _ in range(2):
            torch.manual_seed(0)
            models.append(RelationSheafModel(dd, cfg, graphs, 'static').train())
            optimizers.append(torch.optim.Adam(models[-1].parameters(), lr=.006, weight_decay=.0007))
        with tempfile.TemporaryDirectory() as directory:
            observer = GraphDiagnostics(models[1], Path(directory)/'trace.csv', .0007)
            try:
                for epoch in (1, 2):
                    rng_states = []
                    for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
                        torch.manual_seed(50 + epoch)
                        model.train()
                        if idx:
                            observer.begin(epoch, 2, .006)
                        optimizer.zero_grad(set_to_none=True)
                        output = model(x)
                        loss = sum(t.square().mean() for t in [output[0], *output[1], *output[2]])
                        loss.backward()
                        if idx:
                            observer.after_backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        if idx:
                            observer.before_step(grad_norm)
                        optimizer.step()
                        if idx:
                            observer.after_step()
                        model.eval()
                        with torch.no_grad():
                            model(x)
                        if idx:
                            observer.finish(loss.detach(), loss.detach(), .5)
                        rng_states.append(torch.get_rng_state())
                    self.assertTrue(torch.equal(*rng_states))
                    for key, value in models[0].state_dict().items():
                        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
                report = observer.summary()
                self.assertEqual(report['observations'], 2)
                self.assertGreater(report['first']['value_decay_grad'], 0)
                self.assertGreater(report['first']['value_actual_update_norm'], 0)
                self.assertGreater(report['first']['eval_update_ratio'], 0)
            finally:
                observer.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
