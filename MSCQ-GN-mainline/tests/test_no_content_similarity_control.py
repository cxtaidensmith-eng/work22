"""Synthetic equivalence and training checks; no cohort training or file edits."""
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.no_content_similarity_control import NoContentSimilarityControlModel
from Model.relation_sheaf import RelationSheafModel, modal_candidates, conformal_maps, cooperative_operator
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


REMOVED = 'propagation.edge_query.weight'


def random_states():
    states = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        states.extend(value.clone() for value in torch.cuda.get_rng_state_all())
    return states


def output_probe(output):
    logits, embeddings, auxiliary = output
    return sum((value * torch.linspace(-.71, .93, value.numel(), dtype=value.dtype,
                device=value.device).reshape_as(value)).sum()
               for value in (logits, *embeddings, *auxiliary))


class NoContentSimilarityControlTests(unittest.TestCase):
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

    def model(self, mode='NO_CONTENT_SIMILARITY', device='cpu', dtype=torch.float64, seed=0):
        torch.manual_seed(193 + seed)
        model = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=seed)
                 if mode == 'native' else NoContentSimilarityControlModel(
                     self.dd, self.cfg, self.graphs, mode=mode, training_seed=seed))
        return model.to(device=device, dtype=dtype)

    def assert_rng_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def constructor_check(self, device):
        models, streams = {}, {}
        for mode in ('native', 'C', 'NO_CONTENT_SIMILARITY'):
            models[mode] = self.model(mode, device, torch.float32)
            streams[mode] = random_states()
            self.assert_rng_equal(streams['native'], streams[mode])
        native = models['native']
        for mode in ('C', 'NO_CONTENT_SIMILARITY'):
            candidate = models[mode]
            expected_names = [name for name in native.state_dict()
                              if mode == 'C' or name != REMOVED]
            self.assertEqual(list(candidate.state_dict()), expected_names)
            expected_parameters = [name for name, _ in native.named_parameters()
                                   if mode == 'C' or name != REMOVED]
            self.assertEqual(list(dict(candidate.named_parameters())), expected_parameters)
            for name, actual in candidate.state_dict().items():
                torch.testing.assert_close(actual, native.state_dict()[name], atol=0, rtol=0, msg=name)
            for name in ('edge', 'similarity', 'edge_group'):
                torch.testing.assert_close(getattr(candidate.propagation, name),
                    getattr(native.propagation, name), atol=0, rtol=0)
        p = models['NO_CONTENT_SIMILARITY'].propagation
        self.assertFalse(hasattr(p, 'edge_query'))
        self.assertEqual(p.mode, 'content')
        self.assertEqual(tuple(p.edge_similarity.shape), (6, 6))
        self.assertEqual(tuple(p.edge_bias.shape), (6, 6))
        self.assertEqual(tuple(p.content_id.weight.shape), (6, 16))
        self.assertEqual(p.edge.shape[1], 6 * sum(g['edge'].shape[1] for g in self.graphs))
        self.assertTrue(torch.equal(p.edge[0] // 7, p.edge[1] // 7))
        self.assertEqual(set(p.edge_group.cpu().tolist()), set(range(36)))
        self.assertEqual(sum(p.numel() for p in native.parameters()) -
                         sum(p.numel() for p in models['NO_CONTENT_SIMILARITY'].parameters()), 128)
        self.assertTrue(all(p.requires_grad for p in models['NO_CONTENT_SIMILARITY'].parameters()))

    def compare_zero_query_reference(self, mode, device, training):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        reference = self.model('native', device, dtype)
        candidate = self.model(mode, device, dtype)
        if mode == 'NO_CONTENT_SIMILARITY':
            # Independent reference executes the original unmodified forward:
            # zero normalized queries imply exactly zero content affinity.
            with torch.no_grad():
                reference.propagation.edge_query.weight.zero_()
            reference.propagation.edge_query.weight.requires_grad_(False)
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
        for name, parameter in reference.named_parameters():
            if name == REMOVED and mode == 'NO_CONTENT_SIMILARITY':
                self.assertIsNone(parameter.grad)
                self.assertNotIn(name, other)
                continue
            actual = other[name].grad
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(actual, name)
            self.assertTrue(bool(torch.isfinite(actual).all()), name)
            torch.testing.assert_close(actual, parameter.grad, atol=atol, rtol=rtol, msg=name)

    def test_native_initialization_rng_parameter_order_and_36_paths_cpu(self):
        self.constructor_check('cpu')

    def test_noop_c_forward_and_gradients_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_zero_query_reference('C', 'cpu', training)

    def test_original_zero_query_reference_forward_and_gradients_cpu(self):
        for training in (False, True):
            with self.subTest(training=training):
                self.compare_zero_query_reference('NO_CONTENT_SIMILARITY', 'cpu', training)

    def test_explicit_all_relation_content_paths_and_source_coefficients(self):
        p = self.model().propagation
        with torch.no_grad():
            p.edge_similarity.copy_(torch.linspace(.3, 1.5, 36).reshape(6, 6))
            p.edge_bias.copy_(torch.linspace(-.4, .5, 36).reshape(6, 6))
        tokens = torch.linspace(-1.3, 1.7, 7*6*8, dtype=torch.float64).reshape(7, 6, 8)
        z, values = p.input_norm(tokens), p.value(p.input_norm(tokens))
        raw_maps = p.conditioner(p.content_id(torch.arange(6)))
        paths = []
        for r, graph in enumerate(self.graphs):
            contents = []
            for c in range(6):
                weight = torch.sigmoid(graph['similarity'].double() * p.edge_similarity[r, c]
                                       + p.edge_bias[r, c])
                maps = conformal_maps(raw_maps[c, None].expand(7, -1))
                diffused = cooperative_operator(values[:, c].reshape(7, 2, 4),
                                                graph['edge'], weight, *maps)
                contents.append(diffused.reshape(7, 8))
            paths.append(torch.stack(contents, dim=0))
        delta = torch.stack(paths).mean(0).permute(1, 0, 2)
        expected = p.output_norm(tokens - p.step_logit.sigmoid() * p.output(F.gelu(delta)))
        torch.testing.assert_close(p(tokens), expected, atol=1e-11, rtol=1e-9)
        # The control remains content-dependent through a[r,c] and b[r,c].
        graph = self.graphs[0]
        w0 = torch.sigmoid(graph['similarity'].double()*p.edge_similarity[0, 0]+p.edge_bias[0, 0])
        w1 = torch.sigmoid(graph['similarity'].double()*p.edge_similarity[0, 1]+p.edge_bias[0, 1])
        self.assertGreater(float((w0-w1).abs().max()), 1e-4)

    def test_optimizer_active_training_parameters_and_strict_serialization(self):
        native, model = self.model('native'), self.model()
        optimizer = relation_optimizer(model, self.cfg, 0.)
        expected = optimizer_manifest(native, relation_optimizer(native, self.cfg, 0.))
        actual = optimizer_manifest(model, optimizer)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1]['names'], [n for n in expected[1]['names'] if n != REMOVED])
        self.assertEqual(expected[1]['parameters']-actual[1]['parameters'], 128)
        self.assertEqual(actual[1]['weight_decay'], 0.)
        self.assertEqual(actual[0]['weight_decay'], .0007)
        watched = ('propagation.edge_similarity', 'propagation.edge_bias',
                   'propagation.content_id.weight', 'propagation.conditioner.0.weight',
                   'propagation.conditioner.2.weight')
        before = {name: dict(model.named_parameters())[name].detach().clone() for name in watched}
        for _ in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = output_probe(model(self.x.double()))
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            for name in watched:
                self.assertGreater(float(dict(model.named_parameters())[name].grad.abs().max()), 0., name)
            optimizer.step()
        for name in watched:
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        replica = self.model().eval()
        replica.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(model(self.x.double()), replica(self.x.double()), atol=0, rtol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, replica.state_dict()[name], atol=0, rtol=0)

    def test_tad_shaped_parameter_and_optimizer_budgets(self):
        widths, offset, groups = [138, 150, 3, 36, 24, 9], 0, []
        for width in widths:
            groups.append(list(range(offset, offset+width))); offset += width
        dd = dict(self.dd, Modal_Index=groups, Feature_Num=360, In_Channels=360)
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.Hidden_size, cfg.num_heads, cfg.num_layers, cfg.global_word_emb = 96, 4, 5, 64
        x = torch.randn(7, 360, generator=torch.Generator().manual_seed(141))
        graphs = modal_candidates(x, groups, k=2)
        for mode, budget in (('C', 895096), ('NO_CONTENT_SIMILARITY', 893560)):
            torch.manual_seed(0)
            model = NoContentSimilarityControlModel(dd, cfg, graphs, mode=mode, training_seed=0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), budget)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), budget)
            manifest = optimizer_manifest(model, relation_optimizer(model, cfg, 0.))
            self.assertEqual(manifest[0]['parameters'], 874091)
            self.assertEqual(manifest[1]['parameters'], budget-874091)

    def test_cohort_shape_and_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            NoContentSimilarityControlModel(self.dd, self.cfg, self.graphs, mode='other')
        with self.assertRaises(ValueError):
            self.model().propagation(torch.zeros(6, 6, 8, dtype=torch.float64))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_native_rng_and_zero_query_reference_forward_gradients_cuda(self):
        self.constructor_check('cuda')
        self.compare_zero_query_reference('NO_CONTENT_SIMILARITY', 'cuda', training=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
