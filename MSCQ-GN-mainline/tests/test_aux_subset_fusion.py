"""Verify unchanged prediction path, RNG, label isolation and zero-weight control."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.subset_fusion import SubsetFusionModel
from Model.aux_subset_fusion import AuxiliarySubsetFusionModel, classification_objective


class AuxiliaryTests(unittest.TestCase):
    def check_device(self, device):
        torch.set_num_threads(1)
        x, dd, cfg = tiny_case()
        x = x.to(device)
        chosen = ('COGNITIVE_TEST', 'MRI')
        plain = SubsetFusionModel(dd, cfg, chosen, 0).to(device)
        before = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        auxiliary = AuxiliarySubsetFusionModel(dd, cfg, chosen, 0).to(device)
        self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[1],torch.cuda.get_rng_state_all())))
        for name, state in plain.state_dict().items():
            torch.testing.assert_close(state, auxiliary.state_dict()[name], atol=0, rtol=0)
        for training in (False, True):
            plain.train(training); auxiliary.train(training)
            torch.manual_seed(901)
            expected = plain(x)
            after = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            torch.manual_seed(901)
            actual = auxiliary(x)
            torch.testing.assert_close(actual['logits'], expected['logits'], atol=0, rtol=0)
            torch.testing.assert_close(actual['tokens'], expected['tokens'], atol=0, rtol=0)
            self.assertTrue(torch.equal(after[0],torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(after[1],torch.cuda.get_rng_state_all())))
        # Changing a training-only head cannot affect the inference prediction.
        auxiliary.eval()
        old = auxiliary(x)['logits'].detach()
        with torch.no_grad():
            auxiliary.auxiliary_heads[0].weight.add_(7)
        torch.testing.assert_close(old, auxiliary(x)['logits'], atol=0, rtol=0)
        y = torch.arange(len(x), device=device) % dd['Class_Num']
        fit = torch.arange(len(x), device=device) < 5
        weights = torch.tensor([.8,.6,.4],device=device)
        output = auxiliary(x)
        loss = classification_objective(output,y,fit,weights,.1)
        changed_y = y.clone(); changed_y[~fit] = 999
        alternate = classification_objective(output,changed_y,fit,weights,.1)
        torch.testing.assert_close(loss['total'],alternate['total'],atol=0,rtol=0)
        torch.testing.assert_close(loss['total'],loss['main']+.05*loss['auxiliary'].sum(),atol=0,rtol=0)
        loss['total'].backward()
        for name,p in auxiliary.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        # With lambda=0 the common weights follow the original optimizer path.
        auxiliary = AuxiliarySubsetFusionModel(dd,cfg,chosen,0).to(device)
        plain = SubsetFusionModel(dd,cfg,chosen,0).to(device)
        optimizers = [torch.optim.Adam(m.parameters(),lr=.006,weight_decay=.0007)
                      for m in (plain,auxiliary)]
        for step in range(12):
            for i,model in enumerate((plain,auxiliary)):
                model.train(); optimizers[i].zero_grad(set_to_none=True)
                torch.manual_seed(220+step)
                out = model(x)
                objective = (F.cross_entropy(out['logits'][fit],y[fit],weight=weights,label_smoothing=.05)
                             if i==0 else classification_objective(out,y,fit,weights,0.)['total'])
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizers[i].step()
            for name,state in plain.state_dict().items():
                torch.testing.assert_close(state,auxiliary.state_dict()[name],atol=2e-6,rtol=2e-6,msg=name)
        self.assertEqual(auxiliary.parameter_counts()['inference_parameters'],plain.parameter_counts()['total'])

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
