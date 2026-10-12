"""Independent, synthetic-data checks of the class-readout controls.

These tests never load a cohort or start an experimental training run. Adam
updates are checked against a hand-written coupled-L2 recurrence, and uniform
pooling against the existing zero-QK MultiheadAttention implementation.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.readout_controls import (ReadoutControlModel, readout_optimizer,
    readout_optimizer_manifest, readout_diagnostics, uniform_intervention)
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest
from test_modal_evidence import tiny_case


def streams():
    result = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        result += [v.clone() for v in torch.cuda.get_rng_state_all()]
    return result


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class ReadoutControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x, self.dd, self.cfg = tiny_case(dropout=.37, noise=.1)
        self.cfg.num_layers = 2
        for key, value in dict(global_word_emb=None, semantic_branch='both', adj_mode='learned',
                modal_structure='modal', readout_mode='class_query', lr=.006,
                weight_decay=.0007, grad_clip=1.).items():
            setattr(self.cfg, key, value)
        self.graphs = modal_candidates(self.x, self.dd['Modal_Index'], k=2)

    def model(self, mode, device='cpu', dtype=torch.float32):
        torch.manual_seed(71)
        model = (RelationSheafModel(self.dd, self.cfg, self.graphs, 'content', training_seed=0)
                 if mode == 'original' else
                 ReadoutControlModel(self.dd, self.cfg, self.graphs, mode, training_seed=0))
        return model.to(device=device, dtype=dtype)

    def assert_streams(self, left, right):
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def assert_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0, msg=name)

    def common_model_check(self, device):
        models, states = {}, {}
        for mode in ('original', 'C', 'QK_NODECAY', 'UNIFORM'):
            models[mode] = self.model(mode, device)
            states[mode] = streams()
            self.assert_streams(states['original'], states[mode])
        original_state = clone_state(models['original'])
        for mode in ('C', 'QK_NODECAY'):
            self.assert_state(original_state, clone_state(models[mode]))
        uniform_state = clone_state(models['UNIFORM'])
        for name, value in original_state.items():
            if not name.startswith('label_pools.'):
                torch.testing.assert_close(value, uniform_state[name], atol=0, rtol=0, msg=name)
        hidden, classes = self.cfg.Hidden_size, self.dd['Class_Num']
        removed = classes * (2*hidden*hidden + 3*hidden)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(models['original']), count(models['C']))
        self.assertEqual(count(models['original']), count(models['QK_NODECAY']))
        self.assertEqual(count(models['original'])-count(models['UNIFORM']), removed)
        for pool in models['UNIFORM'].label_pools:
            self.assertFalse(hasattr(pool, 'query'))
            self.assertFalse(hasattr(pool, 'attn'))
            self.assertEqual(set(dict(pool.named_parameters())),
                {'v_proj.weight', 'v_proj.bias', 'out_proj.weight', 'out_proj.bias',
                 'norm_kv.weight', 'norm_kv.bias', 'norm_out.weight', 'norm_out.bias',
                 'ffn.0.weight', 'ffn.0.bias', 'ffn.3.weight', 'ffn.3.bias'})
        zero_qk = copy.deepcopy(models['original'])
        with torch.no_grad():
            for pool in zero_qk.label_pools:
                pool.query.zero_()
                pool.attn.in_proj_weight[:2*hidden].zero_()
                pool.attn.in_proj_bias[:2*hidden].zero_()
        for train in (False, True):
            outputs, rng_after = {}, {}
            for mode in ('original', 'C', 'QK_NODECAY', 'zero_qk', 'UNIFORM'):
                model = zero_qk if mode == 'zero_qk' else models[mode]
                model.train(train)
                torch.manual_seed(331)
                with torch.no_grad():
                    outputs[mode] = model(self.x.to(device))
                rng_after[mode] = streams()
            for mode in ('C', 'QK_NODECAY'):
                torch.testing.assert_close(outputs['original'], outputs[mode], atol=2e-6, rtol=2e-6)
                self.assert_streams(rng_after['original'], rng_after[mode])
            torch.testing.assert_close(outputs['zero_qk'], outputs['UNIFORM'], atol=2e-6, rtol=2e-6)
            self.assert_streams(rng_after['zero_qk'], rng_after['UNIFORM'])

    @staticmethod
    def expected_decay(name, parameter, hidden, mode, decay):
        """Define the experimental claim independently of optimizer internals."""
        weight = torch.full_like(parameter, decay)
        if name.startswith('propagation.') and not name.startswith('propagation.output_norm.'):
            weight.zero_()
        elif mode == 'QK_NODECAY' and name.startswith('label_pools.'):
            if name.endswith('.query'):
                weight.zero_()
            elif name.endswith(('.attn.in_proj_weight', '.attn.in_proj_bias')):
                weight[:2*hidden] = 0
        return weight

    def optimizer_check(self, device):
        # A visible decay is intentional in this tiny optimizer-only check;
        # these settings never enter an experiment configuration.
        cfg = copy.copy(self.cfg)
        cfg.weight_decay, cfg.lr = .23, .017
        for mode in ('C', 'QK_NODECAY', 'UNIFORM'):
            model = self.model(mode, device, torch.float64)
            optimizer = readout_optimizer(model, cfg)
            params = dict(model.named_parameters())
            listed = [p for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(len(listed), len(params))
            self.assertEqual({id(p) for p in listed}, {id(p) for p in params.values()})
            manifest = readout_optimizer_manifest(model, optimizer)
            json.dumps(manifest, allow_nan=False)
            if mode == 'C':
                native = relation_optimizer(model, cfg, 0.)
                self.assertEqual(optimizer_manifest(model, optimizer), optimizer_manifest(model, native))
            reference = {n: p.detach().clone() for n, p in params.items()}
            moment = {n: torch.zeros_like(p) for n, p in params.items()}
            variance = {n: torch.zeros_like(p) for n, p in params.items()}
            masks = {n: self.expected_decay(n, p, self.cfg.Hidden_size, mode, cfg.weight_decay)
                     for n, p in params.items()}
            beta1, beta2, epsilon = .9, .999, 1e-8
            for step in range(1, 4):
                original_grads = {}
                for index, (name, parameter) in enumerate(params.items()):
                    coordinates = torch.arange(parameter.numel(), device=device, dtype=torch.float64).reshape(parameter.shape)
                    gradient = torch.sin(coordinates*.173 + index*.31 + step*.71) * (3+step)
                    parameter.grad = gradient.clone()
                    original_grads[name] = gradient
                # The clipping norm includes data gradients only, BEFORE L2.
                norm = torch.sqrt(sum(g.square().sum() for g in original_grads.values()))
                factor = min(1., .37/(float(norm)+1e-6))
                self.assertLess(factor, 1.)
                torch.nn.utils.clip_grad_norm_(model.parameters(), .37, error_if_nonfinite=True)
                for name, parameter in params.items():
                    clipped = original_grads[name]*factor
                    torch.testing.assert_close(parameter.grad, clipped, atol=1e-12, rtol=1e-12)
                    effective = clipped + masks[name]*reference[name]
                    moment[name] = beta1*moment[name] + (1-beta1)*effective
                    variance[name] = beta2*variance[name] + (1-beta2)*effective.square()
                    mhat = moment[name]/(1-beta1**step)
                    vhat = variance[name]/(1-beta2**step)
                    reference[name] = reference[name] - cfg.lr*mhat/(vhat.sqrt()+epsilon)
                clipped_snapshot = {n: p.grad.detach().clone() for n, p in params.items()}
                optimizer.step()
                for name, parameter in params.items():
                    torch.testing.assert_close(parameter, reference[name], atol=3e-12, rtol=3e-11,
                                               msg=f'{device} {mode} step{step}: {name}')
                    torch.testing.assert_close(optimizer.state[parameter]['exp_avg'], moment[name], atol=1e-12, rtol=1e-11)
                    torch.testing.assert_close(optimizer.state[parameter]['exp_avg_sq'], variance[name], atol=1e-12, rtol=1e-11)
                    torch.testing.assert_close(parameter.grad, clipped_snapshot[name], atol=0, rtol=0)

    @staticmethod
    def retained_pairs(original, uniform):
        hidden = original.attn.embed_dim
        result = [(original.attn.in_proj_weight, uniform.v_proj.weight, slice(2*hidden, None)),
                  (original.attn.in_proj_bias, uniform.v_proj.bias, slice(2*hidden, None)),
                  (original.attn.out_proj.weight, uniform.out_proj.weight, None),
                  (original.attn.out_proj.bias, uniform.out_proj.bias, None)]
        for module in ('norm_kv', 'norm_out', 'ffn'):
            right = dict(getattr(uniform, module).named_parameters())
            for name, parameter in getattr(original, module).named_parameters():
                result.append((parameter, right[name], None))
        return result

    def uniform_layer_check(self, device):
        original = self.model('C', device, torch.float64).label_pools[0]
        uniform = self.model('UNIFORM', device, torch.float64).label_pools[0]
        hidden = self.cfg.Hidden_size
        with torch.no_grad():
            original.query.zero_()
            original.attn.in_proj_weight[:2*hidden].zero_()
            original.attn.in_proj_bias[:2*hidden].zero_()
        self.assertEqual(original.attn.dropout, uniform.attention_dropout)
        pairs = self.retained_pairs(original, uniform)
        for left, right, selection in pairs:
            torch.testing.assert_close(left if selection is None else left[selection], right, atol=0, rtol=0)
        generator = torch.Generator().manual_seed(917)
        tokens = torch.randn(5, 3, hidden, generator=generator, dtype=torch.float64).to(device)
        target = torch.randn(5, hidden, generator=generator, dtype=torch.float64).to(device)
        for train in (False, True):
            original.train(train)
            uniform.train(train)
            original.zero_grad(set_to_none=True)
            uniform.zero_grad(set_to_none=True)
            inputs = [tokens.clone().requires_grad_(True), tokens.clone().requires_grad_(True)]
            values, endings = [], []
            for layer, value in zip((original, uniform), inputs):
                torch.manual_seed(113)
                output = layer(value)
                (output*target).sum().backward()
                values.append(output)
                endings.append(streams())
            torch.testing.assert_close(values[0], values[1], atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=1e-10, rtol=1e-9)
            self.assert_streams(endings[0], endings[1])
            for left, right, selection in pairs:
                self.assertIsNotNone(left.grad)
                self.assertIsNotNone(right.grad)
                gradient = left.grad if selection is None else left.grad[selection]
                torch.testing.assert_close(gradient, right.grad, atol=1e-10, rtol=1e-9)
        # Independent deterministic mean-V equation; no production pool helper.
        original.eval()
        uniform.eval()
        value = F.linear(original.norm_kv(tokens), original.attn.in_proj_weight[2*hidden:],
                         original.attn.in_proj_bias[2*hidden:]).mean(1)
        value = F.linear(value, original.attn.out_proj.weight, original.attn.out_proj.bias)
        expected = value + original.ffn(original.norm_out(value))
        torch.testing.assert_close(uniform(tokens), expected, atol=1e-10, rtol=1e-9)
        # Disable only FFN dropout: remaining stochasticity must be attention dropout.
        for module in uniform.ffn.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.
        uniform.train()
        observations = []
        for seed in (11, 53):
            torch.manual_seed(seed)
            before = streams()
            observations.append(uniform(tokens))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, streams())))
        self.assertGreater(float((observations[0]-observations[1]).abs().max()), 1e-6)

    def passive_diagnostics_check(self, device):
        for mode in ('C', 'QK_NODECAY', 'UNIFORM'):
            model = self.model(mode, device).eval()
            for index, parameter in enumerate(model.parameters()):
                if index % 7 == 0:
                    parameter.grad = torch.full_like(parameter, .17)
            state = clone_state(model)
            buffers = {n: b.clone() for n, b in model.named_buffers()}
            gradients = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
            objects = [id(p) for p in model.label_pools]
            flags = {n: m.training for n, m in model.named_modules()}
            torch.manual_seed(991)
            before = streams()
            diagnostic = readout_diagnostics(model, self.x.to(device))
            self.assertEqual(len(diagnostic['layers']), self.dd['Class_Num'])
            self.assertEqual(diagnostic['mode'], mode)
            json.dumps(diagnostic, allow_nan=False)
            normal, collapsed = uniform_intervention(model, self.x.to(device))
            self.assertEqual(tuple(normal.shape), (7, 3))
            self.assertEqual(tuple(collapsed.shape), (7, 3))
            self.assertTrue(torch.isfinite(normal).all() and torch.isfinite(collapsed).all())
            if mode == 'UNIFORM':
                torch.testing.assert_close(normal, collapsed, atol=2e-6, rtol=2e-6)
                for layer in diagnostic['layers']:
                    self.assertEqual(layer['attention_max_abs_deviation'], 0.)
                    self.assertIsNone(layer['query_norm'])
            self.assert_streams(before, streams())
            self.assert_state(state, clone_state(model))
            self.assert_state(buffers, {n: b for n, b in model.named_buffers()})
            self.assertEqual(objects, [id(p) for p in model.label_pools])
            self.assertEqual(flags, {n: m.training for n, m in model.named_modules()})
            for name, parameter in model.named_parameters():
                if gradients[name] is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(gradients[name], parameter.grad, atol=0, rtol=0)
            # The diagnostic's attention is checked against the original MHA path.
            if mode != 'UNIFORM':
                clone = copy.deepcopy(model)
                with torch.no_grad():
                    h = clone.Feature_Modal(self.x.to(device))
                    gate = clone.modal_gate_logit.sigmoid()
                    tokens = clone.modal_token_encoder(h)*gate.view(1, -1, 1)
                    tokens = clone.propagation(tokens)
                    for block in clone.shared_transformer:
                        tokens = block(tokens)
                    for index, pool in enumerate(clone.label_pools):
                        kv = pool.norm_kv(tokens)
                        _, attention = pool.attn(pool.query.expand(len(tokens), -1, -1), kv, kv,
                                                 need_weights=True, average_attn_weights=False)
                        deviation = float((attention-1/tokens.shape[1]).abs().max())
                        self.assertAlmostEqual(diagnostic['layers'][index]['attention_max_abs_deviation'],
                                               deviation, delta=2e-7)

    def training_diagnostics_rejection_check(self, device):
        model = self.model('C', device)
        for mixed in (False, True):
            model.eval() if mixed else model.train()
            if mixed:
                model.shared_transformer[0].train()
            flags = {n: m.training for n, m in model.named_modules()}
            before = streams()
            state = clone_state(model)
            for function in (readout_diagnostics, uniform_intervention):
                with self.assertRaises(ValueError):
                    function(model, self.x.to(device))
                self.assertEqual(flags, {n: m.training for n, m in model.named_modules()})
                self.assert_streams(before, streams())
                self.assert_state(state, clone_state(model))

    def test_common_initialization_rng_and_full_forward_cpu(self):
        self.common_model_check('cpu')

    def test_coupled_adam_mask_and_clipping_reference_cpu(self):
        self.optimizer_check('cpu')

    def test_uniform_original_mha_outputs_gradients_dropout_cpu(self):
        self.uniform_layer_check('cpu')

    def test_diagnostics_and_intervention_passive_cpu(self):
        self.passive_diagnostics_check('cpu')
        self.training_diagnostics_rejection_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_common_initialization_rng_and_full_forward_cuda(self):
        self.common_model_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_coupled_adam_mask_and_clipping_reference_cuda(self):
        self.optimizer_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_uniform_original_mha_outputs_gradients_dropout_cuda(self):
        self.uniform_layer_check('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_diagnostics_and_intervention_passive_cuda(self):
        self.passive_diagnostics_check('cuda')
        self.training_diagnostics_rejection_check('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
