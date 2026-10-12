"""Synthetic checks for matched-only paths with a learned shared mapping.

The numerical reference is the existing D1 model with its content embedding
rows manually tied. No real cohort training or experiment files are touched.
"""
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.c_mechanism_controls import MechanismControlModel
from Model.matched_shared_mapping_control import MatchedSharedMappingControlModel
from Model.relation_sheaf import modal_candidates, conformal_maps, cooperative_operator
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


EMBEDDING = 'propagation.content_id.weight'


def random_states():
    states = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        states.extend(state.clone() for state in torch.cuda.get_rng_state_all())
    return states


def output_probe(output):
    logits, embeddings, auxiliary = output
    return sum((value * torch.linspace(-.71, .93, value.numel(),
                    dtype=value.dtype, device=value.device).reshape_as(value)).sum()
               for value in (logits, *embeddings, *auxiliary))


class MatchedSharedMappingControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        groups = [[0, 1, 2], [3, 4], [5], [6, 7], [8, 9, 10], [11]]
        self.dd = dict(Modal_Name=['MRI', 'PET', 'CSF', 'RISK_FACTOR',
                                  'COGNITIVE_TEST', 'ROI_AVERAGE'],
                       Modal_Index=groups, Class_Num=3, Feature_Num=12,
                       In_Channels=12, Sample_Num=7, Adj=torch.eye(7))
        self.cfg = SimpleNamespace(Hidden_size=8, Drop_rate=.3, input_noise_std=.1,
            num_heads=2, num_layers=2, ChebGCN_K=3, drop_path=.1, seed=0,
            global_word_emb=8, semantic_branch='both', adj_mode='learned',
            modal_structure='modal', readout_mode='class_query', lr=.006,
            weight_decay=.0007)
        self.x = torch.randn(7, 12, generator=torch.Generator().manual_seed(103))
        self.graphs = modal_candidates(self.x, groups, k=2)

    def model(self, mode, device='cpu', dtype=torch.float64, seed=0):
        torch.manual_seed(193 + seed)
        model = (MechanismControlModel(self.dd, self.cfg, self.graphs, 'D1',
                                       training_seed=seed)
                 if mode == 'native' else MatchedSharedMappingControlModel(
                     self.dd, self.cfg, self.graphs, mode=mode, training_seed=seed))
        return model.to(device=device, dtype=dtype)

    def assert_rng_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def constructor_check(self, device):
        models, streams = {}, {}
        for mode in ('native', 'D1', 'D1_SHARED_MAP'):
            models[mode] = self.model(mode, device, torch.float32)
            streams[mode] = random_states()
            self.assert_rng_equal(streams['native'], streams[mode])
        native = models['native']
        state = native.state_dict()
        for mode in ('D1', 'D1_SHARED_MAP'):
            candidate = models[mode]
            self.assertEqual(list(candidate.state_dict()), list(state))
            self.assertEqual(list(dict(candidate.named_parameters())),
                             list(dict(native.named_parameters())))
            for name, expected in state.items():
                if name == EMBEDDING and mode == 'D1_SHARED_MAP':
                    # The constructor reduces the original FP32 CPU embedding.
                    expected = expected.cpu().mean(0, keepdim=True).to(device)
                torch.testing.assert_close(candidate.state_dict()[name], expected,
                                           atol=0, rtol=0, msg=name)
            for name in ('edge', 'similarity', 'edge_group'):
                torch.testing.assert_close(getattr(candidate.propagation, name),
                    getattr(native.propagation, name), atol=0, rtol=0)
            p = candidate.propagation
            self.assertEqual(p.edge.shape[1], sum(g['edge'].shape[1] for g in self.graphs))
            self.assertEqual(p.edge_similarity.numel(), 6)
            self.assertEqual(p.edge_bias.numel(), 6)
            self.assertTrue(torch.equal(p.edge[0] // 7, p.edge[1] // 7))
            self.assertEqual(int(p.edge.max()) // 7, 5)
        shared = models['D1_SHARED_MAP']
        self.assertEqual(tuple(shared.propagation.content_id.weight.shape), (1, 16))
        self.assertTrue(shared.propagation.content_id.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in native.parameters()) -
                         sum(p.numel() for p in shared.parameters()), 80)
        self.assertTrue(all(p.requires_grad for p in shared.parameters()))

    def compare_outputs_and_gradients(self, mode, device, training):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        reference = self.model('native', device, dtype)
        candidate = self.model(mode, device, dtype)
        if mode == 'D1_SHARED_MAP':
            with torch.no_grad():
                reference.propagation.content_id.weight.copy_(
                    candidate.propagation.content_id.weight.expand_as(
                        reference.propagation.content_id.weight))
        inputs = [self.x.to(device=device, dtype=dtype).clone().requires_grad_(True)
                  for _ in range(2)]
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
            expected = (parameter.grad.sum(0, keepdim=True)
                        if name == EMBEDDING and mode == 'D1_SHARED_MAP' else parameter.grad)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        if mode == 'D1_SHARED_MAP':
            self.assertGreater(float(other[EMBEDDING].grad.abs().max()), 0.)
            rows = reference.propagation.content_id.weight.grad
            self.assertGreater(float((rows - rows.mean(0, keepdim=True)).abs().max()), 0.)

    def test_constructor_cpu_mean_rng_parameter_order_and_buffers(self):
        self.constructor_check('cpu')

    def test_noop_d1_full_output_and_gradients_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_outputs_and_gradients('D1', 'cpu', training)

    def test_shared_tied_d1_reference_gradients_sum_six_rows_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_outputs_and_gradients('D1_SHARED_MAP', 'cpu', training)

    def test_one_graph_per_content_without_dividing_by_six(self):
        p = self.model('D1_SHARED_MAP').propagation
        tokens = torch.linspace(-1.3, 1.7, 7*6*8, dtype=torch.float64).reshape(7, 6, 8)
        z = p.input_norm(tokens)
        q = F.normalize(p.edge_query(z), dim=-1)
        values = p.value(z)
        raw_map = p.conditioner(p.content_id.weight)[0]
        pieces = []
        for n, graph in enumerate(self.graphs):
            i, j = graph['edge']
            weight = torch.sigmoid((q[i, n]*q[j, n]).sum(-1) +
                graph['similarity'].double()*p.edge_similarity[n] + p.edge_bias[n])
            value = values[:, n].reshape(7, 2, 4)
            maps = conformal_maps(raw_map[None].expand(7, -1))
            pieces.append(cooperative_operator(value, graph['edge'], weight, *maps).reshape(7, 8))
        delta = torch.stack(pieces, dim=1)
        expected = p.output_norm(tokens - p.step_logit.sigmoid()*p.output(F.gelu(delta)))
        actual = p(tokens)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
        diluted = p.output_norm(tokens - p.step_logit.sigmoid()*p.output(F.gelu(delta/6)))
        self.assertGreater(float((actual-diluted).abs().max()), 1e-7)

    def test_shared_maps_keep_send_receive_distinct_and_trainable(self):
        layer = self.model('D1_SHARED_MAP').propagation
        ids = torch.arange(6)
        embedded = layer.content_id(ids)
        self.assertEqual(tuple(embedded.shape), (6, 16))
        torch.testing.assert_close(embedded, embedded[0].expand_as(embedded), atol=0, rtol=0)
        with torch.no_grad():
            layer.conditioner[2].bias.add_(torch.tensor([.21, -.13, .17, -.11], dtype=torch.float64))
        parameters = layer.conditioner(layer.content_id(ids))
        torch.testing.assert_close(parameters, parameters[0].expand_as(parameters), atol=0, rtol=0)
        send, receive, _, _ = conformal_maps(parameters[0])
        self.assertGreater(float((send-receive).abs().max()), .01)
        initial_bias = layer.conditioner[2].bias.detach().clone()
        for coordinate in range(4):
            with torch.no_grad():
                layer.conditioner[2].bias.copy_(initial_bias)
                layer.conditioner[2].bias[coordinate] += .06
            changed_send, changed_receive, _, _ = conformal_maps(
                layer.conditioner(layer.content_id(ids))[0])
            if coordinate in (0, 2):
                torch.testing.assert_close(changed_receive, receive, atol=0, rtol=0)
                self.assertGreater(float((changed_send-send).abs().max()), 1e-4)
            else:
                torch.testing.assert_close(changed_send, send, atol=0, rtol=0)
                self.assertGreater(float((changed_receive-receive).abs().max()), 1e-4)
        with torch.no_grad():
            layer.conditioner[2].bias.copy_(initial_bias)
        tokens = torch.randn(7, 6, 8, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(87))
        probe = torch.linspace(-.3, .9, tokens.numel(), dtype=tokens.dtype).reshape_as(tokens)
        (layer(tokens)*probe).sum().backward()
        self.assertTrue(bool(torch.isfinite(layer.conditioner[2].bias.grad).all()))
        self.assertTrue(bool((layer.conditioner[2].bias.grad.abs() > 1e-12).all()))
        self.assertGreater(float(layer.content_id.weight.grad.abs().max()), 0.)

    def test_optimizer_groups_and_synthetic_update_serialization(self):
        native = self.model('native')
        model = self.model('D1_SHARED_MAP')
        optimizer = relation_optimizer(model, self.cfg, 0.)
        expected = optimizer_manifest(native, relation_optimizer(native, self.cfg, 0.))
        actual = optimizer_manifest(model, optimizer)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1]['names'], expected[1]['names'])
        self.assertEqual(expected[1]['parameters']-actual[1]['parameters'], 80)
        self.assertEqual(actual[1]['weight_decay'], 0.)
        self.assertEqual(actual[0]['weight_decay'], .0007)
        self.assertIn(EMBEDDING, actual[1]['names'])
        self.assertIn('propagation.output_norm.weight', actual[0]['names'])
        before = model.propagation.content_id.weight.detach().clone()
        for _ in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = output_probe(model(self.x.double()))
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.propagation.content_id.weight))
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        replica = self.model('D1_SHARED_MAP').eval()
        replica.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(model(self.x.double()), replica(self.x.double()), atol=0, rtol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, replica.state_dict()[name], atol=0, rtol=0)
        for name in ('edge', 'similarity', 'edge_group'):
            torch.testing.assert_close(getattr(model.propagation, name),
                getattr(replica.propagation, name), atol=0, rtol=0)

    def test_tad_shaped_constructor_parameter_and_optimizer_budgets(self):
        widths = [138, 150, 3, 36, 24, 9]
        offset, groups = 0, []
        for width in widths:
            groups.append(list(range(offset, offset+width)))
            offset += width
        dd = dict(self.dd, Modal_Index=groups, Feature_Num=360, In_Channels=360)
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.Hidden_size, cfg.num_heads, cfg.num_layers, cfg.global_word_emb = 96, 4, 5, 64
        x = torch.randn(7, 360, generator=torch.Generator().manual_seed(141))
        graphs = modal_candidates(x, groups, k=2)
        for mode, expected in (('D1', 895036), ('D1_SHARED_MAP', 894956)):
            torch.manual_seed(0)
            model = MatchedSharedMappingControlModel(dd, cfg, graphs, mode=mode, training_seed=0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)
            groups_manifest = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
            self.assertEqual(groups_manifest[0]['parameters'], 874091)
            self.assertEqual(groups_manifest[1]['parameters'], expected-874091)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_constructor_and_tied_reference_forward_gradients_cuda(self):
        self.constructor_check('cuda')
        self.compare_outputs_and_gradients('D1_SHARED_MAP', 'cuda', training=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
