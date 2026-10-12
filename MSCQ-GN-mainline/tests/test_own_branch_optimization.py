"""Verify branch scope, zero-residual initialization, learning and replay."""
import copy
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Model.own_branch_optimization import OwnBranchOptimizationModel, MODES


def rng():
    return [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]


class OwnBranchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x, self.dd, self.cfg = tiny_case()
        for k, v in dict(global_word_emb=64, semantic_branch='both', adj_mode='learned',
                          modal_structure='modal', readout_mode='class_query').items():
            setattr(self.cfg, k, v)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def exercise(self, device):
        models, states = {}, {}
        for mode in ('native', *MODES):
            torch.manual_seed(18)
            models[mode] = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=0)
                if mode == 'native' else OwnBranchOptimizationModel(self.dd, self.cfg, self.graphs, mode)).to(device)
            states[mode] = rng()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(states['native'], states[mode])))
            for name, value in models['native'].state_dict().items():
                torch.testing.assert_close(value, models[mode].state_dict()[name], atol=0, rtol=0)
        base_count = sum(p.numel() for p in models['native'].parameters())
        for mode in MODES:
            self.assertEqual(sum(p.numel() for p in models[mode].parameters())-base_count,
                             16576 if 'FFN' in mode else 0)
        x = self.x.to(device)
        for training in (False, True):
            outputs, traces, streams = {}, {}, {}
            for mode, model in models.items():
                trace = {}
                hooks = [model.modal_token_encoder.register_forward_pre_hook(
                    lambda m, a: trace.update(x=a[0].detach().clone())),
                    model.Global_Message.register_forward_pre_hook(
                    lambda m, a: trace.update(own_input=a[0].detach().clone())),
                    model.propagation.register_forward_hook(
                    lambda m, a, o: trace.update(population=o.detach().clone()))]
                try:
                    torch.manual_seed(33); model.train(training)
                    with torch.no_grad(): outputs[mode] = model(x)
                    streams[mode] = rng()
                finally:
                    for hook in hooks: hook.remove()
                traces[mode] = trace
                gated = mode not in ('NO_GATE', 'NO_GATE_FFN')
                expected = trace['x'] * model.modal_gate_logit.sigmoid()[model.feature_to_modal] if gated else trace['x']
                torch.testing.assert_close(trace['own_input'], expected, atol=0, rtol=0)
                # Sparse CUDA aggregation may differ at float32 round-off level.
                torch.testing.assert_close(traces['native']['population'], trace['population'], atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(outputs['native'][1:], outputs[mode][1:], atol=2e-6, rtol=2e-6)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(streams['native'], streams[mode])))
            for a, b in (('native', 'C'), ('C', 'FFN'), ('NO_GATE', 'NO_GATE_FFN')):
                torch.testing.assert_close(outputs[a], outputs[b], atol=2e-6, rtol=2e-6)
        for mode in MODES:
            model = models[mode].eval()
            captured = {}
            h = model.Global_Message.register_forward_hook(lambda m, a, o: captured.update(own=o))
            try:
                model(x)
                grad = torch.autograd.grad(captured['own'].sum(), model.modal_gate_logit,
                                           allow_unused=True)[0]
                if mode.startswith('NO_GATE'): self.assertIsNone(grad)
                else: self.assertIsNotNone(grad)
            finally: h.remove()
            # Diagnostics and checkpoint reload do not alter model outputs or RNG.
            with torch.no_grad():
                before = model(x)
                model.capture_diagnostics = True
                state = rng(); after = model(x)
                torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(state, rng())))
                other = copy.deepcopy(model)
                other.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(after, other(x), atol=2e-6, rtol=2e-6)
        for mode in ('FFN', 'NO_GATE_FFN'):
            model = models[mode].eval()
            layer = model.Global_Message.token_ffn
            optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0)
            labels = torch.arange(len(x), device=device) % self.dd['Class_Num']
            first = layer[0].weight.detach().clone()
            for _ in range(5):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x)[0], labels)
                self.assertTrue(torch.isfinite(loss)); loss.backward()
                self.assertIsNotNone(model.modal_gate_logit.grad)
                for p in layer.parameters():
                    self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())
                optimizer.step()
            self.assertGreater(float(layer[2].weight.abs().max()), 0)
            self.assertGreater(float((first-layer[0].weight).abs().max()), 0)
            # The same FFN acts independently on each patient/modality token.
            z = torch.randn(7, 6, 64, device=device)
            out = layer(z)
            torch.testing.assert_close(out.flatten(0, 1), layer(z.flatten(0, 1)), atol=1e-6, rtol=1e-6)
            changed = z.clone(); changed[0, 0] += 1
            torch.testing.assert_close(out[1:], layer(changed)[1:], atol=0, rtol=0)
            torch.testing.assert_close(out[0, 1:], layer(changed)[0, 1:], atol=0, rtol=0)

    def test_cpu(self): self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self): self.exercise('cuda')


if __name__ == '__main__': unittest.main(verbosity=2)
