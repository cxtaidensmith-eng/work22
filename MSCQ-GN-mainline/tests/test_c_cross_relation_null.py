"""Checks for matched graph nulls and unchanged C computation/optimization."""
import copy
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.c_cross_relation_null import (build_null_graphs, apply_cross_relation_control,
                                        CrossRelationNullModel)
from Model.relation_sheaf import modal_candidates, RelationSheafModel


def tiny_case():
    generator = torch.Generator().manual_seed(177)
    x = torch.randn(24, 9, generator=generator)
    modal_index = [list(range(3)), list(range(3, 6)), list(range(6, 9))]
    graphs = modal_candidates(x, modal_index, k=3)
    dd = dict(Modal_Name=['MRI', 'PET', 'COGNITIVE_TEST'], Modal_Index=modal_index,
              Class_Num=3, Feature_Num=9, In_Channels=9, Sample_Num=len(x), Adj=torch.eye(len(x)))
    cfg = SimpleNamespace(Hidden_size=8, Drop_rate=.2, ChebGCN_K=3, num_layers=1,
        num_heads=2, input_noise_std=.1, drop_path=.1, global_word_emb=8,
        semantic_branch='both', adj_mode='learned', modal_structure='modal',
        readout_mode='class_query', seed=0)
    return x, graphs, dd, cfg


class CrossRelationNullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_null_invariants_determinism_and_rng_isolation(self):
        x, graphs, _, _ = tiny_case()
        torch.manual_seed(17)
        np.random.seed(18)
        random.seed(19)
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        original = copy.deepcopy(graphs)
        nulls, report = build_null_graphs(graphs, len(x))
        again, report_again = build_null_graphs(graphs, len(x))
        self.assertEqual(report, report_again)
        self.assertTrue(report['all_targets_reached'])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        after_np = np.random.get_state()
        self.assertEqual(after_np[0], numpy_state[0])
        np.testing.assert_array_equal(after_np[1], numpy_state[1])
        self.assertEqual(after_np[2:], numpy_state[2:])
        self.assertEqual(random.getstate(), python_state)
        for relation, (source, before, null, repeated) in enumerate(zip(graphs, original, nulls, again)):
            for key in source:
                self.assertTrue(torch.equal(source[key], before[key]))
            self.assertTrue(torch.equal(null['edge'], repeated['edge']))
            self.assertTrue(torch.equal(null['similarity'], repeated['similarity']))
            self.assertTrue(torch.equal(source['degree'], null['degree']))
            self.assertTrue(torch.equal(source['similarity'].sort().values,
                                        null['similarity'].sort().values))
            edge = null['edge']
            pairs = set(map(tuple, edge.T.tolist()))
            self.assertEqual(len(pairs), edge.shape[1])
            self.assertTrue(all(i != j and (j, i) in pairs for i, j in pairs))
            prior_by_pair = dict(zip(map(tuple, edge.T.tolist()), null['similarity'].tolist()))
            self.assertTrue(all(value == prior_by_pair[(j, i)]
                                for (i, j), value in prior_by_pair.items()))
            item = report['relations'][relation]
            self.assertEqual(item['successful_swaps'], 20 * item['undirected_edge_count'])
            self.assertLess(item['edge_overlap_fraction'], .6)
            self.assertTrue(item['degree_exact'] and item['prior_histogram_exact'])

    def test_bounded_failure_is_explicit_for_complete_graph(self):
        x = torch.eye(4)
        graphs = modal_candidates(x, [[0, 1, 2, 3]], k=3)
        _, report = build_null_graphs(graphs, 4, swaps_per_edge=2, max_attempts_per_swap=3)
        item = report['relations'][0]
        self.assertFalse(report['all_targets_reached'])
        self.assertEqual(item['successful_swaps'], 0)
        self.assertEqual(item['attempts'], item['attempt_limit'])

    def test_diagonal_unchanged_and_single_null_reused_across_contents(self):
        x, graphs, dd, cfg = tiny_case()
        nulls, _ = build_null_graphs(graphs, len(x))
        model = RelationSheafModel(dd, cfg, graphs, 'content')
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        state = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        apply_cross_relation_control(model, graphs, 'XREL_NULL', nulls)
        self.assertEqual(identities, {name: id(p) for name, p in model.named_parameters()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(state), set(model.state_dict()))
        for name, value in state.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        propagation = model.propagation
        for relation in range(3):
            for content in range(3):
                group = relation * 3 + content
                mask = propagation.edge_group == group
                expected = graphs[relation] if relation == content else nulls[relation]
                self.assertTrue(torch.equal(propagation.edge[:, mask] - group * len(x),
                                            expected['edge']))
                self.assertTrue(torch.equal(propagation.similarity[mask], expected['similarity']))
        self.assertEqual({'edge', 'similarity', 'edge_group'}, propagation._non_persistent_buffers_set)
        model.train()
        output = model(x)
        loss = sum(t.square().mean() for t in [output[0], *output[1], *output[2]])
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_identity_intervention_matches_logits_gradients_updates_and_rng(self):
        x, graphs, dd, cfg = tiny_case()
        models, construction_rng = [], []
        for factory in (
                lambda: RelationSheafModel(dd, cfg, graphs, 'content'),
                lambda: CrossRelationNullModel(dd, cfg, graphs, 'C'),
                lambda: CrossRelationNullModel(dd, cfg, graphs, 'XREL_NULL', graphs)):
            torch.manual_seed(47)
            models.append(factory().train())
            construction_rng.append(torch.get_rng_state().clone())
        for candidate, rng in zip(models[1:], construction_rng[1:]):
            self.assertTrue(torch.equal(construction_rng[0], rng))
            self.assertEqual(list(models[0].state_dict()), list(candidate.state_dict()))
            for name, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        optimizers = [torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
                      for model in models]
        for step in range(2):
            outputs, rngs = [], []
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(900 + step)
                output = model(x)
                flat = [output[0], *output[1], *output[2]]
                sum(value.square().mean() for value in flat).backward()
                outputs.append([value.detach().clone() for value in flat])
                rngs.append(torch.get_rng_state().clone())
            for i in (1, 2):
                self.assertTrue(torch.equal(rngs[0], rngs[i]))
                for expected, actual in zip(outputs[0], outputs[i]):
                    self.assertTrue(torch.equal(expected, actual))
                other = dict(models[i].named_parameters())
                for name, parameter in models[0].named_parameters():
                    self.assertTrue(torch.equal(parameter.grad, other[name].grad), name)
            for optimizer in optimizers:
                optimizer.step()
            for candidate in models[1:]:
                for name, value in models[0].state_dict().items():
                    self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)

    def test_validation_rejects_invalid_or_unmatched_graphs(self):
        x, graphs, dd, cfg = tiny_case()
        bad = copy.deepcopy(graphs)
        bad[0]['similarity'][0] += 1
        with self.assertRaisesRegex(ValueError, 'reverse priors'):
            build_null_graphs(bad, len(x))
        model = RelationSheafModel(dd, cfg, graphs, 'content')
        with self.assertRaisesRegex(ValueError, 'requires one explicit null'):
            apply_cross_relation_control(model, graphs, 'XREL_NULL')
        nulls, _ = build_null_graphs(graphs, len(x))
        nulls[0]['similarity'].add_(1)
        with self.assertRaisesRegex(ValueError, 'prior multiset'):
            apply_cross_relation_control(model, graphs, 'XREL_NULL', nulls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
