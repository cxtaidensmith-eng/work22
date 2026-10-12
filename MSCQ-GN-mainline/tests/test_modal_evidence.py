"""Behavioral checks for patient-local additive and pairwise classifiers."""
import inspect
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.modal_evidence import ModalEvidenceModel

MODES = {'A': 'additive', 'B': 'pairwise', 'F': 'fusion'}


def tiny_case(dropout=.3, noise=.1):
    groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    dd = dict(Modal_Name=['MRI', 'COGNITIVE_TEST', 'ROI_AVERAGE'],
              Modal_Index=groups, Class_Num=3, Feature_Num=9,
              In_Channels=9, Sample_Num=7, Adj=torch.eye(7))
    cfg = SimpleNamespace(Hidden_size=8, Drop_rate=dropout,
                          input_noise_std=noise, num_heads=2, num_layers=1,
                          ChebGCN_K=3, drop_path=.1, seed=0)
    generator = torch.Generator().manual_seed(103)
    return torch.randn(7, 9, generator=generator), dd, cfg


class ModalEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.x, self.dd, self.cfg = tiny_case()

    def models(self, device='cpu', cfg=None):
        result = {}
        for mode in ('A', 'B', 'F'):
            torch.manual_seed(9)
            result[mode] = ModalEvidenceModel(
                self.dd, cfg or self.cfg, MODES[mode], training_seed=9, rank=8
            ).to(device)
        return result

    @staticmethod
    def trainable_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def randomize_pair_parameters(self, additive, pairwise):
        common = set(dict(additive.named_parameters()))
        extra = [(name, p) for name, p in pairwise.named_parameters()
                 if name not in common]
        self.assertTrue(extra, 'B must own trainable pair-specific parameters')
        with torch.no_grad():
            for head in pairwise.pair_heads:
                head.output.weight.normal_(mean=0., std=.2)
        return extra

    def test_shared_initialization_and_constructor_cpu_rng(self):
        models, states = {}, []
        for mode in ('A', 'B', 'F'):
            torch.manual_seed(9)
            before = torch.get_rng_state().clone()
            models[mode] = ModalEvidenceModel(self.dd, self.cfg, MODES[mode],
                                              training_seed=9, rank=8)
            states.append(torch.get_rng_state().clone())
            self.assertTrue(torch.equal(before, states[-1]))
        for state in states[1:]:
            self.assertTrue(torch.equal(states[0], state))
        baseline = dict(models['A'].named_parameters())
        for mode in ('B', 'F'):
            candidate = dict(models[mode].named_parameters())
            common = baseline.keys() & candidate.keys()
            self.assertGreater(len(common), 5)
            for name in common:
                torch.testing.assert_close(baseline[name], candidate[name],
                                           atol=0, rtol=0, msg=name)

    def test_initial_additive_pairwise_outputs_and_component_accounting(self):
        models = self.models()
        results = {mode: model.eval()(self.x) for mode, model in models.items()}
        torch.testing.assert_close(results['A']['logits'], results['B']['logits'],
                                   atol=0, rtol=0)
        for mode in ('A', 'B'):
            result = results[mode]
            self.assertEqual(tuple(result['unimodal_logits'].shape), (7, 3, 3))
            expected = result['unimodal_logits'].sum(1)
            if mode == 'B':
                self.assertEqual(tuple(result['pair_logits'].shape), (7, 3, 3))
                expected = expected + result['pair_logits'].sum(1)
            else:
                self.assertIsNone(result['pair_logits'])
            torch.testing.assert_close(result['logits'], expected, atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(results['B']['pair_logits'])), 0)
        for model in models.values():
            self.assertNotIn('labels', inspect.signature(model.forward).parameters)
            self.assertNotIn('label', inspect.signature(model.forward).parameters)
            with self.assertRaises(TypeError):
                model(self.x, labels=torch.arange(7) % 3)

    def test_unimodal_scores_depend_only_on_own_modality_and_patient(self):
        models = self.models()
        for mode in ('A', 'B'):
            model = models[mode].eval()
            reference = model(self.x)['unimodal_logits']
            for changed_modality, indices in enumerate(self.dd['Modal_Index']):
                changed = self.x.clone()
                changed[2, indices] += torch.arange(1, len(indices)+1) * 3.
                actual = model(changed)['unimodal_logits']
                other_modalities = [m for m in range(3) if m != changed_modality]
                torch.testing.assert_close(actual[:, other_modalities],
                                           reference[:, other_modalities], atol=0, rtol=0)
                other_patients = [i for i in range(7) if i != 2]
                torch.testing.assert_close(actual[other_patients], reference[other_patients],
                                           atol=0, rtol=0)
                self.assertGreater(float((actual[2, changed_modality] -
                                          reference[2, changed_modality]).abs().max()), 1e-7)

    def test_pair_scores_depend_only_on_the_two_modalities_and_patient(self):
        models = self.models()
        model = models['B'].eval()
        self.randomize_pair_parameters(models['A'], model)
        pairs = list(itertools.combinations(range(3), 2))
        self.assertEqual(tuple(pairs), model.pair_indices)
        reference = model(self.x)
        torch.testing.assert_close(reference['logits'],
            reference['unimodal_logits'].sum(1) + reference['pair_logits'].sum(1),
            atol=0, rtol=0)
        for changed_modality, indices in enumerate(self.dd['Modal_Index']):
            changed = self.x.clone()
            changed[2, indices] += torch.arange(1, len(indices)+1) * 3.
            actual = model(changed)['pair_logits']
            other_patients = [i for i in range(7) if i != 2]
            torch.testing.assert_close(actual[other_patients],
                reference['pair_logits'][other_patients], atol=0, rtol=0)
            for pair_index, pair in enumerate(pairs):
                difference = actual[2, pair_index] - reference['pair_logits'][2, pair_index]
                if changed_modality not in pair:
                    torch.testing.assert_close(difference, torch.zeros_like(difference),
                                               atol=0, rtol=0)
                else:
                    self.assertGreater(float(difference.abs().max()), 1e-7)

    def test_classification_training_reaches_pair_factors_and_encoder(self):
        models = self.models(cfg=tiny_case(dropout=0., noise=0.)[2])
        model = models['B'].train()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradient_seen = {name: False for name in before}
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        target = torch.arange(7) % 3
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            result = model(self.x)
            loss = F.cross_entropy(result['logits'], target)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
                gradient_seen[name] |= bool(torch.count_nonzero(p.grad))
            optimizer.step()
        for name, p in model.named_parameters():
            self.assertTrue(gradient_seen[name], name)
            self.assertTrue(bool(torch.isfinite(p).all()), name)
            self.assertFalse(torch.equal(before[name], p.detach()), name)

    def test_fusion_budget_matches_pairwise_and_all_parameters_are_active(self):
        models = self.models(cfg=tiny_case(dropout=0., noise=0.)[2])
        pair_count = self.trainable_count(models['B'])
        fusion_count = self.trainable_count(models['F'])
        one_hidden_unit = 3 * self.cfg.Hidden_size + self.dd['Class_Num'] + 1
        self.assertLessEqual(abs(pair_count-fusion_count), max(pair_count*.01, one_hidden_unit))
        model = models['F'].train()
        F.cross_entropy(model(self.x)['logits'], torch.arange(7) % 3).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        counts = model.parameter_counts()
        self.assertIsInstance(counts, dict)
        self.assertEqual(fusion_count, counts['total'])
        self.assertEqual(counts['total'], counts['common'] + counts['heads'])

    def test_input_noise_is_active_without_dropout_and_preserves_clean_modality(self):
        cfg = tiny_case(dropout=0., noise=.1)[2]
        models = self.models(cfg=cfg)
        samples = []
        for seed in (401, 402):
            modes = []
            for model in models.values():
                model.train()
                torch.manual_seed(seed)
                modes.append(model(self.x)['tokens'])
            for other in modes[1:]:
                torch.testing.assert_close(modes[0], other, atol=0, rtol=0)
            samples.append(modes[0])
        torch.testing.assert_close(samples[0][:, 1], samples[1][:, 1], atol=0, rtol=0)
        for modality in (0, 2):
            self.assertFalse(torch.equal(samples[0][:, modality], samples[1][:, modality]))

    def test_training_noise_and_shared_dropout_match_all_variants(self):
        models = self.models()
        outputs, rng_after = [], []
        for model in models.values():
            model.train()
            torch.manual_seed(311)
            outputs.append(model(self.x))
            rng_after.append(torch.get_rng_state().clone())
        for result, state in zip(outputs[1:], rng_after[1:]):
            torch.testing.assert_close(outputs[0]['tokens'], result['tokens'], atol=0, rtol=0)
            self.assertTrue(torch.equal(rng_after[0], state))
        torch.testing.assert_close(outputs[0]['logits'], outputs[1]['logits'], atol=0, rtol=0)
        torch.manual_seed(312)
        different = models['A'](self.x)
        self.assertFalse(torch.equal(outputs[0]['tokens'], different['tokens']))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA RNG check requires CUDA')
    def test_cuda_rng_and_shared_training_tokens_match(self):
        constructor_states, forward_states, tokens = [], [], []
        for mode in ('A', 'B', 'F'):
            torch.manual_seed(9)
            model = ModalEvidenceModel(self.dd, self.cfg, MODES[mode],
                                       training_seed=9, rank=8).cuda().train()
            constructor_states.append(torch.cuda.get_rng_state_all())
            torch.manual_seed(311)
            tokens.append(model(self.x.cuda())['tokens'].detach())
            forward_states.append(torch.cuda.get_rng_state_all())
        for collection in (constructor_states, forward_states):
            for state in collection[1:]:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(collection[0], state)))
        for other in tokens[1:]:
            torch.testing.assert_close(tokens[0], other, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
