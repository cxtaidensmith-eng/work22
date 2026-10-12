"""Behavioral and dense-reference checks for sparse joint evidence routing."""
import copy
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.joint_evidence_routing import JointEvidenceRouting, JointEvidenceRoutingModel
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from test_modal_evidence import tiny_case


def empty_graphs(modalities):
    return [dict(edge=torch.empty(2, 0, dtype=torch.long)) for _ in range(modalities)]


def test_graphs():
    # Only source modality 1 of patient 1 may contribute to patient 0.
    graphs = empty_graphs(3)
    graphs[1]['edge'] = torch.tensor([[0, 1], [1, 0]])
    return graphs


def rng_state():
    return (torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()])


class JointEvidenceRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(120)

    def assert_rng_equal(self, left, right):
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertEqual(len(left[1]), len(right[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left[1], right[1])))

    @staticmethod
    def dense_reference(layer, tokens):
        # Compute only eligible scores inside each row, independently of the
        # sparse segment normalization and custom sparse multiplication.
        query, key, value = layer.projected_evidence(tokens)
        rows = []
        for index in range(len(query)):
            columns = layer.edge[1, layer.edge[0] == index]
            modality = columns.remainder(layer.modalities)
            modality[columns == len(query)] = layer.modalities
            scores = (query[index] * key[columns]).sum(-1) / math.sqrt(layer.query_size)
            scores = scores + layer.modal_bias[index % layer.modalities, modality]
            rows.append((scores.softmax(0)[:, None] * value[columns]).sum(0))
        aggregated = torch.stack(rows).reshape_as(tokens)
        return layer.output_norm(tokens + layer.step_logit.sigmoid() * layer.output(aggregated))

    def check_device(self, device):
        sparse = JointEvidenceRouting(8, 3, 4, test_graphs(), 'joint').double().to(device)
        dense = copy.deepcopy(sparse)
        tokens = torch.randn(4, 3, 8, device=device, dtype=torch.float64, requires_grad=True)
        dense_tokens = tokens.detach().clone().requires_grad_(True)
        before = rng_state()
        output = sparse(tokens)
        self.assert_rng_equal(before, rng_state())
        expected = self.dense_reference(dense, dense_tokens)
        torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-11)
        objective = torch.randn_like(output)
        (output * objective).sum().backward()
        (expected * objective).sum().backward()
        torch.testing.assert_close(tokens.grad, dense_tokens.grad, atol=1e-10, rtol=1e-9)
        for (name, parameter), (other_name, other) in zip(sparse.named_parameters(), dense.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-10, rtol=1e-9, msg=name)

        weights, _ = sparse.attention(tokens.detach())
        totals = weights.new_zeros(12).index_add(0, sparse.edge[0], weights)
        torch.testing.assert_close(totals, torch.ones_like(totals), atol=1e-12, rtol=1e-12)
        diagnostics = sparse.attention_diagnostics(tokens.detach())
        self.assertAlmostEqual(sum(diagnostics[k]['mean'] for k in ('own', 'neighbor', 'null')), 1., 12)
        self.assertGreater(diagnostics['null']['mean'], 0.)
        self.assertGreater(diagnostics['neighbor_cross']['mean'], 0.)

        # Own and joint share every parameter; empty support makes the entire
        # forward/backward computation equal, not just parameter counts.
        own = JointEvidenceRouting(8, 3, 4, test_graphs(), 'own').double().to(device)
        own.load_state_dict(sparse.state_dict(), strict=True)
        joint_empty = JointEvidenceRouting(8, 3, 4, empty_graphs(3), 'joint').double().to(device)
        joint_empty.load_state_dict(own.state_dict(), strict=True)
        a = tokens.detach().clone().requires_grad_(True)
        b = tokens.detach().clone().requires_grad_(True)
        own_output, empty_output = own(a), joint_empty(b)
        # CUDA sparse reductions can differ at the last floating-point bit.
        torch.testing.assert_close(own_output, empty_output, atol=1e-12, rtol=1e-12)
        (own_output * objective).sum().backward()
        (empty_output * objective).sum().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-11)
        for (name, p), (_, q) in zip(own.named_parameters(), joint_empty.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-11, rtol=1e-11, msg=name)

        base = tokens.detach()
        perturbation = torch.arange(8, device=device, dtype=base.dtype).square()
        eligible = base.clone()
        eligible[1, 1] += perturbation
        ineligible = base.clone()
        ineligible[1, 0] += perturbation
        other_patient = base.clone()
        other_patient[2] += perturbation
        torch.testing.assert_close(own(base)[0], own(eligible)[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(sparse(base)[0], sparse(ineligible)[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(sparse(base)[0], sparse(other_patient)[0], atol=1e-12, rtol=1e-12)
        self.assertGreater(float((sparse(base)[0, 0] - sparse(eligible)[0, 0]).abs().max()), 1e-7)
        own_cross = base.clone()
        own_cross[0, 1] += perturbation
        self.assertGreater(float((own(base)[0, 0] - own(own_cross)[0, 0]).abs().max()), 1e-7)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')

    def test_model_common_initialization_and_rng(self):
        x, dd, cfg = tiny_case()
        cfg.global_word_emb = None
        cfg.semantic_branch = 'both'
        cfg.adj_mode = 'learned'
        cfg.modal_structure = 'modal'
        cfg.readout_mode = 'class_query'
        graphs = modal_candidates(x, dd['Modal_Index'], k=2)
        models, streams = {}, {}
        for mode in ('content', 'own', 'joint'):
            torch.manual_seed(400)
            factory = RelationSheafModel if mode == 'content' else JointEvidenceRoutingModel
            models[mode] = factory(dd, cfg, graphs, mode, training_seed=0)
            streams[mode] = rng_state()
        self.assert_rng_equal(streams['content'], streams['own'])
        self.assert_rng_equal(streams['content'], streams['joint'])
        own, joint = models['own'].state_dict(), models['joint'].state_dict()
        self.assertEqual(own.keys(), joint.keys())
        for name, parameter in own.items():
            torch.testing.assert_close(parameter, joint[name], atol=0, rtol=0, msg=name)
        for name, parameter in models['content'].state_dict().items():
            if not name.startswith('propagation.'):
                torch.testing.assert_close(parameter, own[name], atol=0, rtol=0, msg=name)
        for mode in ('own', 'joint'):
            model = models[mode].eval()
            output = model(x)
            self.assertIsInstance(output, tuple)
            self.assertEqual(len(output), 3)
            self.assertEqual(tuple(output[0].shape), (len(x), dd['Class_Num']))
        # All stochastic layers outside routing see the same shapes and consume
        # the same CPU/CUDA training stream in the two complete models.
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            training_streams = []
            for mode in ('own', 'joint'):
                model = models[mode].to(device).train()
                torch.manual_seed(501)
                model(x.to(device))
                training_streams.append(rng_state())
            self.assert_rng_equal(*training_streams)


if __name__ == '__main__':
    unittest.main()
