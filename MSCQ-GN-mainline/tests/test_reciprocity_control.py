"""Behavior and gradient checks for reciprocity, using synthetic data only."""
import io
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from test_shared_mapping_control import output_probe
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Model.reciprocity_control import ReciprocityControlModel, relation_marks
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


class ReciprocityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x, self.dd, self.cfg = tiny_case()
        for name, value in dict(num_layers=2, global_word_emb=8, semantic_branch='both',
                adj_mode='learned', modal_structure='modal', readout_mode='class_query',
                lr=.006, weight_decay=.0007).items():
            setattr(self.cfg, name, value)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def model(self, mode, device='cpu'):
        torch.manual_seed(981)
        cls = RelationSheafModel if mode == 'native' else ReciprocityControlModel
        actual_mode = 'content' if mode == 'native' else mode
        return cls(self.dd, self.cfg, self.graphs, actual_mode, training_seed=0).to(
            device=device, dtype=torch.float64 if device == 'cpu' else torch.float32)

    def test_known_reciprocal_labels_and_private_rng(self):
        neighbors = torch.tensor([[1], [0], [1], [2]])
        selected = torch.zeros(4, 4, dtype=torch.bool).scatter_(1, neighbors, True)
        graph = dict(neighbors=neighbors, edge=(selected|selected.T).nonzero().T)
        before = torch.get_rng_state().clone()
        code, stats = relation_marks(graph, 4)
        expected = torch.tensor([1., 1., 0., 0., 0., 0.]) - 1/3
        torch.testing.assert_close(code, expected)
        self.assertEqual(stats['mutual_edges'], 1)
        self.assertEqual(stats['unilateral_edges'], 2)
        shuffled, other = relation_marks(graph, 4, True, 3)
        self.assertEqual(stats, other)
        self.assertTrue(torch.equal(torch.sort(code).values, torch.sort(shuffled).values))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def zero_equivalence(self, device):
        native = self.model('native', device)
        original_rng = torch.get_rng_state().clone()
        original = dict(native.named_parameters())
        for mode in ('C', 'REAL', 'SHUFFLED'):
            model = self.model(mode, device)
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
            for name, value in original.items():
                torch.testing.assert_close(dict(model.named_parameters())[name], value, atol=0, rtol=0)
            extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in native.parameters())
            self.assertEqual(extra, 0 if mode == 'C' else 9)
            for training in (False, True):
                outputs, inputs, grads = [], [], []
                for item in (native, model):
                    item.zero_grad(set_to_none=True); item.train(training)
                    x = self.x.to(device=device, dtype=next(item.parameters()).dtype).clone().requires_grad_()
                    torch.manual_seed(671)
                    output = item(x); output_probe(output).backward()
                    outputs.append(output); inputs.append(x.grad.clone())
                    grads.append({n: p.grad.clone() for n, p in item.named_parameters() if n in original})
                tol = dict(atol=1e-11, rtol=1e-9) if device == 'cpu' else dict(atol=3e-5, rtol=3e-4)
                torch.testing.assert_close(outputs[0], outputs[1], **tol)
                torch.testing.assert_close(inputs[0], inputs[1], **tol)
                for name in original:
                    torch.testing.assert_close(grads[0][name], grads[1][name], msg=name, **tol)
                if mode != 'C':
                    gradient = model.propagation.reciprocity_bias.grad
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    self.assertGreater(float(gradient.abs().max()), 0.)

    def test_zero_initialization_function_and_gradient_cpu(self):
        self.zero_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_zero_initialization_function_and_gradient_cuda(self):
        self.zero_equivalence('cuda:0')

    def test_nonzero_symmetric_weights_formula_and_serialization(self):
        model = self.model('REAL').eval()
        layer = model.propagation
        with torch.no_grad():
            layer.reciprocity_bias.copy_(torch.linspace(-.7, .9, 9).reshape(3, 3))
        z = torch.randn(7, 3, 8, dtype=torch.float64)
        weights = layer.edge_weights(z)
        i, j = layer.edge
        pairs = {tuple(pair): k for k, pair in enumerate(layer.edge.T.tolist())}
        reverse = torch.tensor([pairs[(b, a)] for a, b in layer.edge.T.tolist()])
        # The inherited float32 cosine GEMM can differ by roundoff on reverse
        # entries. Preserve those native priors; the NEW metadata is exact.
        torch.testing.assert_close(layer.reciprocity_code, layer.reciprocity_code[reverse], atol=0, rtol=0)
        adjustment = layer.reciprocity_bias.tanh().flatten()[layer.edge_group] * layer.reciprocity_code
        torch.testing.assert_close(adjustment, adjustment[reverse], atol=0, rtol=0)
        torch.testing.assert_close(weights, weights[reverse], atol=1e-7, rtol=1e-7)
        q = torch.nn.functional.normalize(layer.edge_query(z), dim=-1)
        offset = 0
        for r, graph in enumerate(self.graphs):
            code, _ = relation_marks(graph, 7)
            for m in range(3):
                a, b = graph['edge']; size = len(a)
                expected = torch.sigmoid((q[a,m]*q[b,m]).sum(-1)
                    + graph['similarity'].double()*layer.edge_similarity[r,m] + layer.edge_bias[r,m]
                    + layer.reciprocity_bias[r,m].tanh()*code.double())
                torch.testing.assert_close(weights[offset:offset+size], expected)
                offset += size
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = self.model('REAL').eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(model(self.x.double()), restored(self.x.double()), atol=0, rtol=0)

    def test_shuffle_capacity_counts_and_optimizer(self):
        real, random = self.model('REAL'), self.model('SHUFFLED')
        self.assertEqual(real.reciprocity_statistics, random.reciprocity_statistics)
        self.assertFalse(torch.equal(real.propagation.reciprocity_code, random.propagation.reciprocity_code))
        for graph in (real, random):
            manifest = optimizer_manifest(graph, relation_optimizer(graph, self.cfg, 0.))
            self.assertIn('propagation.reciprocity_bias', manifest[1]['names'])
            self.assertEqual(manifest[1]['weight_decay'], 0.)
            self.assertEqual(manifest[0]['weight_decay'], .0007)
        self.assertEqual(optimizer_manifest(real, relation_optimizer(real, self.cfg, 0.)),
                         optimizer_manifest(random, relation_optimizer(random, self.cfg, 0.)))


if __name__ == '__main__':
    unittest.main()
