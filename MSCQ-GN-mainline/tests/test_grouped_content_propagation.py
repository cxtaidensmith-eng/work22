"""Independent checks of C's channel-group transport extension.

All optimization checks use tiny synthetic tensors, never cohort training.
The dense reference below does not call the production map/diffusion helpers.
"""
import copy
import inspect
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.grouped_content_propagation import GroupedContentModel, GroupedContentPropagation
from Model.relation_sheaf import RelationContentPropagation, RelationSheafModel, modal_candidates
from test_modal_evidence import tiny_case


def rng_states():
    states = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        states.extend(s.clone() for s in torch.cuda.get_rng_state_all())
    return states


def dense_group_reference(values, edge, weights, parameters):
    """Dense normalized two-pass diffusion, one original channel pair at a time."""
    nodes, _, channels = values.shape
    groups = parameters.shape[1]
    width = channels // groups
    adjacency = values.new_zeros(nodes, nodes).index_put(
        (edge[0], edge[1]), weights, accumulate=True)
    degree = adjacency.sum(-1)
    outputs = []
    for channel in range(channels):
        raw = parameters[:, channel // width]
        angle = math.pi * raw[:, :2].tanh()
        scales = 2 * raw[:, 2:].sigmoid()
        # Build send/receive explicitly; rows remain the original paired axes.
        matrices = []
        for endpoint in range(2):
            c, s = angle[:, endpoint].cos(), angle[:, endpoint].sin()
            matrix = torch.stack((torch.stack((c, -s), -1),
                                  torch.stack((s, c), -1)), -2)
            matrices.append(matrix * scales[:, endpoint, None, None])
        send, receive = matrices
        current = values[:, :, channel:channel+1]
        for endpoint in range(2):
            diagonal = degree * scales[:, endpoint].square()
            normalizer = (1 + diagonal).rsqrt()[:, None, None]
            normalized = current * normalizer
            outgoing = torch.matmul(send, normalized)
            incoming = (adjacency @ outgoing.squeeze(-1)).unsqueeze(-1)
            received = torch.matmul(receive.transpose(-1, -2), incoming)
            current = (diagonal[:, None, None] * normalized - received) * normalizer
        outputs.append(current)
    return torch.cat(outputs, -1)


class GroupedContentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(419)
        self.x, self.dd, self.cfg = tiny_case()
        self.cfg.num_layers = 2
        for key, value in dict(global_word_emb=None, semantic_branch='both',
                               adj_mode='learned', modal_structure='modal',
                               readout_mode='class_query').items():
            setattr(self.cfg, key, value)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def base(self, device='cpu', dtype=torch.float64):
        torch.manual_seed(73)
        return RelationContentPropagation(8, 3, 7, self.graphs, 'content').to(
            device=device, dtype=dtype)

    def assert_rng_equal(self, before, after):
        self.assertEqual(len(before), len(after))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def constructor_and_map_check(self, device):
        base = self.base(device)
        z = torch.randn(7, 3, 8, dtype=torch.float64, device=device)
        before = rng_states()
        layers = {name: GroupedContentPropagation(copy.deepcopy(base), groups=g,
                         conditioner_width=w, training_seed=0)
                  for name, g, w in (('G1', 1, 32), ('G2', 2, 32), ('W38', 1, 38))}
        self.assert_rng_equal(before, rng_states())
        count = sum(p.numel() for p in base.parameters())
        for name, delta in (('G1', 0), ('G2', 132), ('W38', 126)):
            layer = layers[name]
            self.assertEqual(sum(p.numel() for p in layer.parameters()), count+delta)
            self.assertEqual(set(base.state_dict()), set(layer.state_dict()))
            for key, value in base.state_dict().items():
                if not key.startswith('conditioner.') or name == 'G1':
                    torch.testing.assert_close(value, layer.state_dict()[key], atol=0, rtol=0)
            for key in ('edge', 'similarity', 'edge_group'):
                torch.testing.assert_close(getattr(base, key), getattr(layer, key), atol=0, rtol=0)
            expected = base.map_values(z).unsqueeze(-2).expand(3, 3, 7, layer.groups, 4)
            torch.testing.assert_close(layer.map_values(z), expected, atol=1e-14, rtol=1e-12)
            torch.testing.assert_close(layer.group_map_parameters(), expected[0, :, 0],
                                       atol=1e-14, rtol=1e-12)
        for group in range(2):
            torch.testing.assert_close(layers['G2'].conditioner[2].weight[4*group:4*(group+1)],
                                       base.conditioner[2].weight, atol=0, rtol=0)
        wide = layers['W38']
        torch.testing.assert_close(wide.conditioner[0].weight[:32], base.conditioner[0].weight,
                                   atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(wide.conditioner[2].weight[:, 32:])), 0)
        self.assertGreater(float(wide.conditioner[0].weight[32:].abs().max()), 0.)

    def layer_equivalence_check(self, device):
        original = self.base(device)
        g1 = GroupedContentPropagation(copy.deepcopy(original), groups=1)
        g2 = GroupedContentPropagation(copy.deepcopy(original), groups=2)
        wide = GroupedContentPropagation(copy.deepcopy(original), conditioner_width=38)
        inputs = [torch.randn(7, 3, 8, dtype=torch.float64, device=device, requires_grad=True)]
        inputs += [inputs[0].detach().clone().requires_grad_(True) for _ in range(3)]
        target = torch.randn_like(inputs[0])
        outputs = [layer(value) for layer, value in zip((original, g1, g2, wide), inputs)]
        # CPU runs use the identical original computation for G1.
        torch.testing.assert_close(outputs[0], outputs[1], atol=0 if device == 'cpu' else 1e-12,
                                   rtol=0 if device == 'cpu' else 1e-12)
        for output in outputs[2:]:
            torch.testing.assert_close(outputs[0], output, atol=1e-12, rtol=1e-12)
        for output in outputs:
            (output * target).sum().backward()
        for value in inputs[1:]:
            torch.testing.assert_close(inputs[0].grad, value.grad, atol=1e-10, rtol=1e-9)
        for name, p in original.named_parameters():
            for layer in (g1, g2, wide):
                q = dict(layer.named_parameters())[name]
                self.assertIsNotNone(p.grad, name)
                self.assertIsNotNone(q.grad, name)
                observed = q.grad
                if layer is g2 and name.startswith('conditioner.2.'):
                    observed = observed.reshape(2, *p.shape).sum(0)
                elif layer is wide and name.startswith('conditioner.0.'):
                    observed = observed[:32]
                elif layer is wide and name == 'conditioner.2.weight':
                    observed = observed[:, :32]
                torch.testing.assert_close(p.grad, observed, atol=1e-10, rtol=1e-9, msg=name)
        # Independent groups have distinct gradients even when their maps start equal.
        gradient = g2.conditioner[2].weight.grad.reshape(2, 4, 32)
        self.assertGreater(float((gradient[0]-gradient[1]).abs().max()), 1e-7)
        # The wide control is initially equivalent but its extra units are learnable.
        self.assertGreater(float(wide.conditioner[2].weight.grad[:, 32:].abs().max()), 1e-7)
        self.assertEqual(int(torch.count_nonzero(wide.conditioner[0].weight.grad[32:])), 0)
        with torch.no_grad():
            wide.conditioner[2].weight[:, 32:].fill_(.01)
        wide.zero_grad(set_to_none=True)
        (wide(inputs[-1].detach()) * target).sum().backward()
        self.assertGreater(float(wide.conditioner[0].weight.grad[32:].abs().max()), 1e-9)

    def dense_operator_check(self, device):
        layer = GroupedContentPropagation(self.base(device), groups=2)
        nodes = 3*3*7
        values = torch.randn(nodes, 2, 4, dtype=torch.float64, device=device, requires_grad=True)
        parameters = (.3*torch.randn(nodes, 2, 4, dtype=torch.float64, device=device)).requires_grad_(True)
        # Symmetric weights retain the production cooperative operator's assumptions.
        row, col = layer.edge
        weights = (.5+.25*(row+col).to(torch.float64).sin()).requires_grad_(True)
        expected_inputs = [v.detach().clone().requires_grad_(True) for v in (values, weights, parameters)]
        actual = layer.grouped_diffusion(values, weights, parameters)
        expected = dense_group_reference(expected_inputs[0], layer.edge, expected_inputs[1],
                                        expected_inputs[2])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        target = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual*target).sum(), (values, weights, parameters))
        expected_grads = torch.autograd.grad((expected*target).sum(), expected_inputs)
        for observed, reference in zip(actual_grads, expected_grads):
            self.assertTrue(bool(torch.isfinite(observed).all()))
            self.assertGreater(float(observed.abs().max()), 1e-7)
            torch.testing.assert_close(observed, reference, atol=1e-10, rtol=1e-9)
        # Pair (h_a,h_{a+H/2}) remains intact; changing group0 cannot alter group1.
        changed = parameters.detach().clone()
        changed[:, 0] += changed.new_tensor([.2, -.1, .3, -.2])
        other = layer.grouped_diffusion(values.detach(), weights.detach(), changed)
        torch.testing.assert_close(other[:, :, 2:], actual.detach()[:, :, 2:],
                                   atol=0 if device == 'cpu' else 1e-12,
                                   rtol=0 if device == 'cpu' else 1e-12)
        self.assertGreater(float((other[:, :, :2]-actual.detach()[:, :, :2]).abs().max()), 1e-4)
        # Distinct maps must not silently average into a common transform.
        collapsed = changed.mean(1, keepdim=True).expand_as(changed)
        collapsed_output = layer.grouped_diffusion(values.detach(), weights.detach(), collapsed)
        self.assertGreater(float((other-collapsed_output).abs().max()), 1e-4)

    def full_model_check(self, device):
        models, streams = {}, {}
        for variant in ('original', 'C', 'G1', 'G2', 'W38'):
            torch.manual_seed(96)
            models[variant] = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=0)
                               if variant == 'original' else GroupedContentModel(
                                   self.dd, self.cfg, self.graphs, variant, training_seed=0)).to(device)
            streams[variant] = rng_states()
            self.assert_rng_equal(streams['original'], streams[variant])
            for name, tensor in models['original'].state_dict().items():
                if not name.startswith('propagation.conditioner.') or variant in ('C', 'G1'):
                    torch.testing.assert_close(tensor, models[variant].state_dict()[name], atol=0, rtol=0)
        x = self.x.to(device)
        for training in (False, True):
            outputs, streams = {}, {}
            for variant, model in models.items():
                torch.manual_seed(238)
                model.train(training)
                outputs[variant] = model(x)
                streams[variant] = rng_states()
                torch.testing.assert_close(outputs['original'], outputs[variant], atol=2e-6, rtol=2e-6)
                self.assert_rng_equal(streams['original'], streams[variant])
            if device == 'cpu':
                torch.testing.assert_close(outputs['original'], outputs['G1'], atol=0, rtol=0)
        for variant in ('G2', 'W38'):
            model = models[variant].eval()
            model.zero_grad(set_to_none=True)
            logits, embeddings, auxiliary = model(x)
            loss = logits.square().sum() + sum(a.square().sum() for a in auxiliary)
            loss.backward()
            for name, p in model.named_parameters():
                if name.startswith('propagation.'):
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            self.assertGreater(float(model.propagation.conditioner[2].weight.grad.abs().max()), 0.)
            self.assertNotIn('labels', inspect.signature(model.forward).parameters)
            with self.assertRaises(TypeError):
                model(x, labels=torch.arange(7))

    def diagnostics_check(self, device):
        model = GroupedContentModel(self.dd, self.cfg, self.graphs, 'G2').to(device).eval()
        with torch.no_grad():
            model.propagation.conditioner[2].bias[4:] += torch.tensor([.2, -.15, .1, -.2], device=device)
            baseline = model(self.x.to(device))
            state = {k: v.clone() for k, v in model.state_dict().items()}
            before = rng_states()
            model.capture_diagnostics = True
            observed = model(self.x.to(device))
            self.assert_rng_equal(before, rng_states())
            torch.testing.assert_close(observed, baseline, atol=2e-6, rtol=2e-6)
            self.assertGreater(model.last_diagnostics['mean_group_parameter_std'], .01)
            for key, value in state.items():
                torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
            normal_maps = model.propagation.group_map_parameters()
            model.propagation.intervention = 'collapse'
            maps = model.propagation.group_map_parameters()
            torch.testing.assert_close(maps, normal_maps.mean(1, keepdim=True).expand_as(normal_maps),
                                       atol=0, rtol=0)
            collapsed = model(self.x.to(device))
            self.assertEqual(model.last_diagnostics['mean_group_parameter_std'], 0.)
            self.assertGreater(float((collapsed[0]-baseline[0]).abs().max()), 1e-7)
            model.propagation.intervention = 'normal'
            torch.testing.assert_close(model(self.x.to(device)), baseline, atol=2e-6, rtol=2e-6)
            replica = copy.deepcopy(model)
            replica.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(replica(self.x.to(device)), baseline, atol=2e-6, rtol=2e-6)

    def test_constructor_and_parameter_budget_cpu(self):
        self.constructor_and_map_check('cpu')

    def test_g1_exact_g2_and_wide_initial_equivalence_cpu(self):
        self.layer_equivalence_check('cpu')

    def test_group_pairing_dense_value_and_gradient_reference_cpu(self):
        self.dense_operator_check('cpu')

    def test_full_model_training_rng_and_initial_equivalence_cpu(self):
        self.full_model_check('cpu')

    def test_diagnostics_and_collapse_cpu(self):
        self.diagnostics_check('cpu')

    def test_invalid_groups_and_shapes(self):
        for groups in (0, 3):
            with self.assertRaises(ValueError):
                GroupedContentPropagation(self.base(), groups=groups)
        with self.assertRaises(ValueError):
            GroupedContentPropagation(self.base(), groups=2, conditioner_width=38)
        layer = GroupedContentPropagation(self.base(), groups=2)
        with self.assertRaises(ValueError):
            layer(torch.zeros(8, 3, 8, dtype=torch.float64))
        with self.assertRaises(ValueError):
            layer.grouped_diffusion(torch.zeros(63, 4, 2), torch.ones(layer.edge.shape[1]),
                                    torch.zeros(63, 2, 4))
        with self.assertRaises(ValueError):
            layer.grouped_diffusion(torch.zeros(63, 2, 4), torch.ones(layer.edge.shape[1]),
                                    torch.zeros(63, 4))
        layer.intervention = 'unknown'
        with self.assertRaises(ValueError):
            layer.group_map_parameters()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_constructor_and_layer_equivalence_cuda(self):
        self.constructor_and_map_check('cuda')
        self.layer_equivalence_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_dense_reference_pairing_and_gradients_cuda(self):
        self.dense_operator_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_model_rng_and_diagnostics_cuda(self):
        self.full_model_check('cuda')
        self.diagnostics_check('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
