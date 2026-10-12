"""C equivalence, gradient-preserving reuse, and schedule/RNG controls."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.alternating_cooperative import AlternatingCooperativeModel, propagate
from Model.relation_sheaf import RelationContentPropagation, RelationSheafModel, modal_candidates
from test_modal_evidence import tiny_case
from analyze_alternating_cooperative import interaction


def rng():
    return [torch.get_rng_state().clone(), *[s.clone() for s in torch.cuda.get_rng_state_all()]]


class AlternatingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        self.x, self.dd, self.cfg = tiny_case()
        self.cfg.num_layers = 5
        for key, value in dict(global_word_emb=None, semantic_branch='both', adj_mode='learned',
                               modal_structure='modal', readout_mode='class_query').items():
            setattr(self.cfg, key, value)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def assert_rng(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def operator_check(self, device):
        original = RelationContentPropagation(8, 3, 7, self.graphs, 'content').double().to(device)
        wrapped = copy.deepcopy(original)
        a = torch.randn(7, 3, 8, device=device, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        target = torch.randn_like(a)
        y = original(a)
        z, w, _ = propagate(wrapped, b)
        torch.testing.assert_close(y, z, atol=1e-12, rtol=1e-12)
        (y*target).sum().backward(); (z*target).sum().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-10)
        for (name, p), (_, q) in zip(original.named_parameters(), wrapped.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9, msg=name)
        # Cached weights must preserve the first-state gradient while current
        # values must still respond to the second input independently.
        a = a.detach().requires_grad_(True)
        current = torch.randn_like(a, requires_grad=True)
        _, w0, _ = propagate(wrapped, a)
        out, w1, _ = propagate(wrapped, current, w0)
        self.assertIs(w0, w1)
        dw, da, db = torch.autograd.grad((out*target).sum(), (w0, a, current), retain_graph=True)
        for grad in (dw, da, db):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.abs().max()), 1e-10)
        with torch.no_grad():
            altered, _, _ = propagate(wrapped, current + torch.randn_like(current), w0)
            self.assertGreater(float((out-altered).abs().max()), 1e-4)
            repeat, wr, _ = propagate(wrapped, a)
            _, wf, _ = propagate(wrapped, current)
            torch.testing.assert_close(wr, w0, atol=0, rtol=0)
            self.assertGreater(float((wf-w0).abs().max()), 1e-6)
            # Edge symmetry is required by the retained cooperative operator.
            edge = wrapped.edge.cpu()
            reverse = {(int(i), int(j)): k for k, (i, j) in enumerate(edge.T)}
            indices = torch.tensor([reverse[int(j), int(i)] for i, j in edge.T], device=device)
            # Candidate cosines originate in float32; conversion to double
            # does not remove the original GEMM roundoff (~1e-8 here).
            torch.testing.assert_close(w0, w0[indices], atol=1e-7, rtol=0)

    def model_check(self, device):
        models, states = {}, {}
        for mode in ('old', 'C', 'N0', 'N1', 'A0', 'A1'):
            torch.manual_seed(12)
            models[mode] = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=0)
                            if mode == 'old' else AlternatingCooperativeModel(
                                self.dd, self.cfg, self.graphs, mode, training_seed=0)).to(device)
            states[mode] = rng()
            self.assert_rng(states['old'], states[mode])
            self.assertEqual(models['old'].state_dict().keys(), models[mode].state_dict().keys())
            for name, parameter in models['old'].state_dict().items():
                torch.testing.assert_close(parameter, models[mode].state_dict()[name], atol=0, rtol=0)
        x = self.x.to(device)
        outputs, streams = {}, {}
        for mode, model in models.items():
            torch.manual_seed(75)
            outputs[mode] = model(x)
            streams[mode] = rng()
            self.assert_rng(streams['old'], streams[mode])
        for left, right in zip(outputs['old'], outputs['C']):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
        for mode in ('old', 'C'):
            output = outputs[mode]
            objective = output[0].square().sum() + sum(t.square().sum() for t in output[2])
            objective.backward()
        for (name, p), (_, q) in zip(models['old'].named_parameters(), models['C'].named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)
        for mode in ('N0', 'N1', 'A0', 'A1'):
            model = models[mode].eval()
            events, weights = [], []
            def traced(layer, tokens, weight=None, diagnostics=False):
                events.append('P')
                result = propagate(layer, tokens, weight, diagnostics)
                weights.append(result[1])
                return result
            handles = [block.register_forward_pre_hook(
                lambda module, args, i=i: events.append(f'F{i+1}'))
                for i, block in enumerate(model.shared_transformer)]
            with torch.no_grad(), patch('Model.alternating_cooperative.propagate', traced):
                plain = model(x)
            for handle in handles:
                handle.remove()
            expected = (['P','F1','F2','P','F3','F4','F5'] if mode.startswith('A')
                        else ['P','P','F1','F2','F3','F4','F5'])
            self.assertEqual(events, expected)
            self.assertEqual(weights[0] is weights[1], mode.endswith('0'))
            model.capture_diagnostics = True
            before = rng()
            with torch.no_grad():
                diagnostic = model(x)
            self.assert_rng(before, rng())
            torch.testing.assert_close(plain, diagnostic, atol=0, rtol=0)
            trace = model.last_diagnostics
            self.assertEqual(trace['reused_same_weight_tensor'], mode.endswith('0'))
            if mode.endswith('0'):
                self.assertEqual(trace['mean_absolute_edge_weight_change'], 0.)
            else:
                self.assertGreater(trace['mean_absolute_edge_weight_change'], 0.)
            model.capture_diagnostics = False
            model.train(); model.zero_grad(set_to_none=True)
            output = model(x)
            (output[0].square().sum()+sum(t.square().sum() for t in output[2])).backward()
            for name, p in model.named_parameters():
                if name.startswith('propagation.'):
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)

    def test_operator_cpu(self):
        self.operator_check('cpu')

    def test_model_cpu(self):
        self.model_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_operator_cuda(self):
        self.operator_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_model_cuda(self):
        self.model_check('cuda')

    def test_paired_interaction_uses_common_folds(self):
        records = {m: {} for m in ('N0','N1','A0','A1')}
        for fold, values in ((1,(.7,.8,.6,.9)), (2,(.8,.6,.8,.7))):
            for mode, value in zip(records, values):
                records[mode][fold] = dict(record={s:dict(acc=value) for s in ('best','last')})
        records['A1'][3] = dict(record={s:dict(acc=0.) for s in ('best','last')})
        result = interaction(records)
        self.assertEqual(result['matched_folds'], [1,2])
        self.assertAlmostEqual(result['stages']['best']['mean_acc_delta'], .15, places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
