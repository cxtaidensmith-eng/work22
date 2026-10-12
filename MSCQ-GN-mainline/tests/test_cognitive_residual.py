"""Verify initialization, direct-path input isolation and trainable supplement."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.subset_fusion import SubsetFusionModel
from Model.cognitive_residual import CognitiveResidualModel


class CognitiveResidualTests(unittest.TestCase):
    def check_device(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        x = x.to(device)
        cognitive = SubsetFusionModel(dd, cfg, ('COGNITIVE_TEST',), 0).to(device)
        before = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        model = CognitiveResidualModel(dd, cfg, 'MRI', 0).to(device)
        self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[1],torch.cuda.get_rng_state_all())))
        for training in (False, True):
            cognitive.train(training); model.train(training)
            torch.manual_seed(1701)
            expected = cognitive(x)['logits']
            after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            torch.manual_seed(1701)
            result = model(x)
            torch.testing.assert_close(result['logits'], expected, atol=2e-6, rtol=2e-6)
            self.assertEqual(int(torch.count_nonzero(result['pair_logits'])), 0)
            self.assertTrue(torch.equal(after[0],torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(after[1],torch.cuda.get_rng_state_all())))
        optimizer = torch.optim.Adam(model.parameters(),lr=.006,weight_decay=.0007)
        y = torch.arange(len(x),device=device) % dd['Class_Num']
        for step in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(x)
            torch.testing.assert_close(output['logits'], output['unimodal_logits'][:,0]+output['pair_logits'][:,0], atol=0, rtol=0)
            loss = torch.nn.functional.cross_entropy(output['logits'][:5],y[:5],label_smoothing=.05)
            loss.backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            if step > 0:
                self.assertGreater(float(model.encoders[0][0].weight.grad.abs().sum()),0)
                self.assertGreater(float(model.head[0].weight.grad.abs().sum()),0)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        original = model(x)
        self.assertGreater(float(original['pair_logits'].abs().sum()),0)
        changed = x.clone(); changed[:,dd['Modal_Index'][0]] += 17
        torch.testing.assert_close(original['unimodal_logits'], model(changed)['unimodal_logits'],atol=0,rtol=0)
        self.assertGreater(float((original['pair_logits']-model(changed)['pair_logits']).abs().sum()),0)
        changed = x.clone(); changed[:,dd['Modal_Index'][2]] += 111
        torch.testing.assert_close(original['logits'],model(changed)['logits'],atol=0,rtol=0)
        changed=x.clone(); changed[0]+=9
        torch.testing.assert_close(original['logits'][1:],model(changed)['logits'][1:],atol=0,rtol=0)
        joint=SubsetFusionModel(dd,cfg,('MRI','COGNITIVE_TEST'),0)
        self.assertEqual(model.parameter_counts()['total'],joint.parameter_counts()['total']+model.parameter_counts()['cognitive_head'])

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__=='__main__':
    unittest.main()
