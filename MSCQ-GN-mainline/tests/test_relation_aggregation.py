"""Equivalence, axes, gradient, RNG, label isolation and diagnostic checks."""
import copy
import inspect
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Model.relation_aggregation import RelationAggregationModel


def rng():
    return [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]


class AggregationTests(unittest.TestCase):
    def check(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        for k, v in dict(global_word_emb=None, semantic_branch='both', adj_mode='learned',
                         modal_structure='modal', readout_mode='class_query').items():
            setattr(cfg, k, v)
        graphs = modal_candidates(x, dd['Modal_Index'], k=2)
        models = {}; streams = {}
        for mode in ('old', 'C', 'R1', 'R2'):
            torch.manual_seed(18)
            models[mode] = (RelationSheafModel(dd,cfg,graphs,'content',training_seed=0)
                           if mode=='old' else RelationAggregationModel(dd,cfg,graphs,mode)).to(device)
            streams[mode] = rng()
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(streams['old'],streams[mode])))
            for name, p in models['old'].state_dict().items():
                torch.testing.assert_close(p,models[mode].state_dict()[name],atol=0,rtol=0)
            for name in ('edge','similarity','edge_group'):
                torch.testing.assert_close(getattr(models['old'].propagation,name),
                                           getattr(models[mode].propagation,name),atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in models['R1'].parameters()),
                         sum(p.numel() for p in models['R2'].parameters()))
        for name, p in models['R1'].state_dict().items():
            torch.testing.assert_close(p,models['R2'].state_dict()[name],atol=0,rtol=0)
        x = x.to(device)
        for training in (False, True):
            values = {}; streams = {}
            for mode, model in models.items():
                torch.manual_seed(31); model.train(training)
                values[mode] = model(x); streams[mode] = rng()
                torch.testing.assert_close(values['old'],values[mode],atol=2e-6,rtol=2e-6)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(streams['old'],streams[mode])))
        # At zero score, added selection derivatives into the backbone are zero.
        for model in models.values():
            model.eval(); model.zero_grad(set_to_none=True)
            logits, emb, aux = model(x)
            (logits.square().sum()+sum(t.square().sum() for t in aux)).backward()
        for name, p in models['old'].named_parameters():
            for mode in ('C','R1','R2'):
                q = dict(models[mode].named_parameters())[name]
                if p.grad is None: self.assertIsNone(q.grad)
                else: torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-5)
        for mode in ('R1','R2'):
            layer = models[mode].propagation
            self.assertGreater(float(layer.relation_score[-1].weight.grad.abs().max()),0.)
            # Independent manual reduction checks axis semantics after nonuniform scoring.
            with torch.no_grad(): layer.relation_score[-1].weight.normal_(0,.3)
            n,m,h = 7,3,8
            z = torch.randn(n,m,h,device=device,requires_grad=True)
            updates = torch.randn(m,m,n,h,device=device,requires_grad=True)
            d = updates.permute(2,0,1,3)
            scores = layer.relation_score(torch.cat((F.layer_norm(z,(h,))[:,None].expand_as(d),
                          F.layer_norm(d,(h,))),-1)).squeeze(-1)+layer.relation_bias[None]
            expected = (scores.mean(0,keepdim=True).expand_as(scores) if mode=='R1' else scores).softmax(1)
            layer.capture_diagnostics = True
            actual = layer.aggregate(z,updates)
            torch.testing.assert_close(layer.last_weights,expected,atol=0,rtol=0)
            torch.testing.assert_close(actual,(expected[...,None]*d).sum(1),atol=0,rtol=0)
            torch.testing.assert_close(expected.sum(1),torch.ones(n,m,device=device))
            if mode=='R1': self.assertEqual(float(expected.std(0,unbiased=False).max()),0.)
            else: self.assertGreater(float(expected.std(0,unbiased=False).max()),1e-5)
            actual.square().sum().backward()
            self.assertGreater(float(z.grad.abs().max()),0.)
            self.assertGreater(float(updates.grad.abs().max()),0.)
            self.assertGreater(float(layer.relation_score[0].weight.grad.abs().max()),0.)
            permutation = torch.tensor([3,1,6,0,2,5,4],device=device)
            torch.testing.assert_close(layer.aggregate(z[permutation],updates[:,:,permutation]),actual[permutation])
            layer.intervention='patient_mean'
            torch.testing.assert_close(layer.aggregate(z,updates),
                (expected.mean(0,keepdim=True)[...,None]*d).sum(1))
            layer.intervention='uniform'
            torch.testing.assert_close(layer.aggregate(z,updates),updates.mean(0).permute(1,0,2))
            layer.intervention='normal'
            self.assertNotIn('labels',inspect.signature(layer.forward).parameters)
            with self.assertRaises(TypeError): layer(z,labels=torch.zeros(n))
            model = models[mode].eval()
            model.capture_diagnostics=False
            with torch.no_grad(): baseline=model(x)
            before=rng(); model.capture_diagnostics=True
            with torch.no_grad(): observed=model(x)
            torch.testing.assert_close(baseline,observed,atol=2e-6,rtol=2e-6)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,rng())))
            replica=copy.deepcopy(model)
            replica.load_state_dict(model.state_dict(),strict=True)
            # CUDA sparse reductions can differ at float32 rounding precision.
            with torch.no_grad(): torch.testing.assert_close(model(x),replica(x),atol=2e-6,rtol=2e-6)

    def test_cpu(self): self.check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self): self.check('cuda')


if __name__=='__main__': unittest.main(verbosity=2)
