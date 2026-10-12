"""Independent shared-embedding control checks on synthetic inputs only.

The numerical reference is native C with its modality embedding rows manually
tied. No cohort training, experiment outputs, or original sources are changed.
"""
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.relation_sheaf import RelationSheafModel, conformal_maps, modal_candidates
from Model.shared_mapping_control import SharedMappingControlModel
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from test_modal_evidence import tiny_case


EMBEDDING = 'propagation.content_id.weight'


def random_states():
    result = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        result.extend(x.clone() for x in torch.cuda.get_rng_state_all())
    return result


def output_probe(output):
    """A nonconstant linear probe reaches all returned representations."""
    logits, embeddings, auxiliary = output
    return sum((tensor * torch.linspace(-.71, .93, tensor.numel(),
                device=tensor.device, dtype=tensor.dtype).reshape_as(tensor)).sum()
               for tensor in (logits, *embeddings, *auxiliary))


class SharedMappingControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(407)
        self.x, self.dd, self.cfg = tiny_case()
        for key, value in dict(num_layers=2, global_word_emb=8,
                semantic_branch='both', adj_mode='learned', modal_structure='modal',
                readout_mode='class_query', lr=.006, weight_decay=.0007).items():
            setattr(self.cfg, key, value)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def model(self, mode, device='cpu', dtype=torch.float64, seed=0):
        torch.manual_seed(193+seed)
        model = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=seed)
                 if mode == 'native' else SharedMappingControlModel(
                     self.dd, self.cfg, self.graphs, mode=mode, training_seed=seed))
        return model.to(device=device, dtype=dtype)

    def assert_rng_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def constructor_check(self, device):
        models, streams = {}, {}
        for mode in ('native', 'C', 'SHARED_MAP'):
            models[mode] = self.model(mode, device=device, dtype=torch.float32)
            streams[mode] = random_states()
            self.assert_rng_equal(streams['native'], streams[mode])
        native_state = models['native'].state_dict()
        for mode in ('C', 'SHARED_MAP'):
            state = models[mode].state_dict()
            self.assertEqual(set(state), set(native_state))
            self.assertEqual(list(dict(models[mode].named_parameters())),
                             list(dict(models['native'].named_parameters())))
            for name, value in native_state.items():
                # Construction happens on CPU before .to(device); preserve that
                # reduction order rather than comparing against a CUDA mean.
                expected = (value.detach().cpu().mean(0, keepdim=True).to(value.device)
                            if name == EMBEDDING and mode == 'SHARED_MAP' else value)
                torch.testing.assert_close(state[name], expected, atol=0, rtol=0, msg=name)
            for name in ('edge', 'similarity', 'edge_group'):
                torch.testing.assert_close(getattr(models[mode].propagation, name),
                    getattr(models['native'].propagation, name), atol=0, rtol=0)
        shared = models['SHARED_MAP']
        self.assertEqual(tuple(shared.propagation.content_id.weight.shape), (1, 16))
        self.assertTrue(shared.propagation.content_id.weight.requires_grad)
        self.assertEqual(shared.propagation.conditioner[0].in_features, 16)
        self.assertEqual(shared.propagation.conditioner[0].out_features, 32)
        self.assertEqual(shared.propagation.conditioner[2].out_features, 4)
        original_count = sum(p.numel() for p in models['native'].parameters())
        self.assertEqual(sum(p.numel() for p in models['C'].parameters()), original_count)
        self.assertEqual(sum(p.numel() for p in shared.parameters()),
                         original_count-16*(len(self.dd['Modal_Index'])-1))
        self.assertTrue(all(p.requires_grad for p in shared.parameters()))

    def compare_outputs_and_gradients(self, mode, device, training):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        reference = self.model('native', device, dtype)
        candidate = self.model(mode, device, dtype)
        if mode == 'SHARED_MAP':
            with torch.no_grad():
                reference.propagation.content_id.weight.copy_(
                    candidate.propagation.content_id.weight.expand_as(reference.propagation.content_id.weight))
        inputs = [self.x.to(device=device, dtype=dtype).clone().requires_grad_(True) for _ in range(2)]
        outputs, streams = [], []
        for model, x in zip((reference, candidate), inputs):
            torch.manual_seed(4103)
            model.train(training)
            outputs.append(model(x))
            streams.append(random_states())
            output_probe(outputs[-1]).backward()
        self.assert_rng_equal(*streams)
        atol, rtol = (1e-11, 1e-9) if device == 'cpu' else (3e-5, 3e-4)
        torch.testing.assert_close(outputs[0], outputs[1], atol=atol, rtol=rtol)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=atol, rtol=rtol)
        self.assertGreater(float(inputs[1].grad.abs().max()), 0.)
        other = dict(candidate.named_parameters())
        self.assertEqual(set(other), set(dict(reference.named_parameters())))
        for name, parameter in reference.named_parameters():
            actual = other[name].grad
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(actual, name)
            self.assertTrue(bool(torch.isfinite(actual).all()), name)
            expected = parameter.grad.sum(0, keepdim=True) if name == EMBEDDING and mode == 'SHARED_MAP' else parameter.grad
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        if mode == 'SHARED_MAP':
            self.assertGreater(float(other[EMBEDDING].grad.abs().max()), 0.)
            # The tied-row reference receives distinct per-modality gradients;
            # the shared model must accumulate them rather than pick one row.
            rows = reference.propagation.content_id.weight.grad
            self.assertGreater(float((rows-rows.mean(0, keepdim=True)).abs().max()), 0.)

    def maps_check(self, device):
        model = self.model('SHARED_MAP', device)
        layer = model.propagation
        m, n, h = layer.modalities, layer.samples, layer.hidden
        z = torch.linspace(-1., 1., n*m*h, dtype=torch.float64, device=device).reshape(n, m, h)
        parameters = layer.map_values(z)
        self.assertEqual(tuple(parameters.shape), (m, m, n, 4))
        torch.testing.assert_close(parameters, parameters[0, 0, 0].expand_as(parameters), atol=0, rtol=0)
        torch.testing.assert_close(parameters, layer.map_values(z.square()+3), atol=0, rtol=0)
        self.assertTrue(parameters.requires_grad)
        # A native model with unequal original content embeddings is not a
        # valid function-equivalent reference; parameter tying is intentional.
        native = self.model('native', device)
        native_parameters = native.propagation.map_values(z)
        self.assertGreater(float((native_parameters[:, 0]-native_parameters[:, 1]).abs().max()), 0.)

    def test_constructor_rng_common_parameters_and_budget_cpu(self):
        self.constructor_check('cpu')

    def test_noop_c_full_output_input_and_parameter_gradients_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_outputs_and_gradients('C', 'cpu', training)

    def test_shared_native_tied_embedding_equivalence_and_gradient_sum_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_outputs_and_gradients('SHARED_MAP', 'cpu', training)

    def test_shared_maps_across_relation_content_patient_cpu(self):
        self.maps_check('cpu')

    def test_sender_and_receiver_are_distinct_trainable_roles(self):
        layer = self.model('SHARED_MAP').propagation
        z = torch.linspace(-.8, 1.1, 7*3*8, dtype=torch.float64).reshape(7, 3, 8)
        with torch.no_grad():
            layer.conditioner[2].bias.add_(torch.tensor([.21, -.13, .17, -.11], dtype=torch.float64))
        raw = layer.map_values(z)[0, 0, 0]
        send, receive, _, _ = conformal_maps(raw)
        self.assertGreater(float((send-receive).abs().max()), .01)
        original_bias = layer.conditioner[2].bias.detach().clone()
        for coordinate in range(4):
            with torch.no_grad():
                layer.conditioner[2].bias.copy_(original_bias)
                layer.conditioner[2].bias[coordinate] += .06
            altered_send, altered_receive, _, _ = conformal_maps(layer.map_values(z)[0, 0, 0])
            if coordinate in (0, 2):
                torch.testing.assert_close(altered_receive, receive, atol=0, rtol=0)
                self.assertGreater(float((altered_send-send).abs().max()), 1e-4)
            else:
                torch.testing.assert_close(altered_send, send, atol=0, rtol=0)
                self.assertGreater(float((altered_receive-receive).abs().max()), 1e-4)
        with torch.no_grad():
            layer.conditioner[2].bias.copy_(original_bias)
        probe = torch.linspace(-.3, .9, z.numel(), dtype=z.dtype).reshape_as(z)
        (layer(z)*probe).sum().backward()
        self.assertTrue(bool(torch.isfinite(layer.conditioner[2].bias.grad).all()))
        self.assertTrue(bool((layer.conditioner[2].bias.grad.abs() > 1e-10).all()))
        self.assertGreater(float(layer.content_id.weight.grad.abs().max()), 0.)

    def test_optimizer_graph_decay_zero_output_norm_original_decay(self):
        native = self.model('native')
        shared = self.model('SHARED_MAP')
        original = optimizer_manifest(native, relation_optimizer(native, self.cfg, 0.))
        observed = optimizer_manifest(shared, relation_optimizer(shared, self.cfg, 0.))
        self.assertEqual(observed[0], original[0])
        self.assertEqual(observed[1]['names'], original[1]['names'])
        self.assertEqual(original[1]['parameters']-observed[1]['parameters'], 32)
        self.assertEqual(observed[1]['weight_decay'], 0.)
        self.assertEqual(observed[1]['lr'], self.cfg.lr)
        self.assertIn(EMBEDDING, observed[1]['names'])
        self.assertIn('propagation.conditioner.2.weight', observed[1]['names'])
        for name in ('propagation.output_norm.weight', 'propagation.output_norm.bias'):
            self.assertIn(name, observed[0]['names'])
            self.assertNotIn(name, observed[1]['names'])
        self.assertEqual(observed[0]['weight_decay'], .0007)

    def test_state_serialization_roundtrip_without_files(self):
        for mode in ('C', 'SHARED_MAP'):
            original = self.model(mode).eval()
            with torch.no_grad():
                original.propagation.content_id.weight.add_(.017)
                original.propagation.conditioner[2].bias.add_(.09)
            buffer = io.BytesIO()
            torch.save(original.state_dict(), buffer)
            buffer.seek(0)
            replica = self.model(mode).eval()
            replica.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            torch.testing.assert_close(original(self.x.double()), replica(self.x.double()), atol=0, rtol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, replica.state_dict()[name], atol=0, rtol=0)

    def test_tad_shaped_synthetic_constructor_exact_parameter_counts(self):
        widths = [138, 150, 3, 36, 24, 9]
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1]+width)
        dd = dict(Modal_Name=['MRI', 'PET', 'CSF', 'RISK_FACTOR', 'COGNITIVE_TEST', 'ROI_AVERAGE'],
                  Modal_Index=[list(range(a, b)) for a, b in zip(offsets[:-1], offsets[1:])],
                  Class_Num=3, Feature_Num=360, In_Channels=360, Sample_Num=7, Adj=torch.eye(7))
        self.cfg.Hidden_size, self.cfg.num_heads, self.cfg.num_layers = 96, 4, 5
        self.cfg.global_word_emb = 64
        generator = torch.Generator().manual_seed(141)
        x = torch.randn(7, 360, generator=generator)
        graphs = modal_candidates(x, dd['Modal_Index'], k=2)
        for mode, expected in (('C', 895096), ('SHARED_MAP', 895016)):
            torch.manual_seed(0)
            model = SharedMappingControlModel(dd, self.cfg, graphs, mode=mode, training_seed=0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_constructor_rng_and_shared_map_invariance_cuda(self):
        self.constructor_check('cuda')
        self.maps_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_noop_and_tied_reference_forward_gradients_cuda(self):
        for mode in ('C', 'SHARED_MAP'):
            with self.subTest(mode=mode):
                self.compare_outputs_and_gradients(mode, 'cuda', training=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
