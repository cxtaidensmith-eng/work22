import sys
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel
from Model.late_patient_readout import LatePatientReadout, LateReadoutModel


class LateReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x,self.graphs,self.dd,self.cfg=tiny_case()

    def test_dense_reference_and_cross_patient_dependency(self):
        torch.manual_seed(12)
        a=LatePatientReadout(8,7,self.graphs,'population').double()
        b=LatePatientReadout(8,7,self.graphs,'self').double(); b.load_state_dict(a.state_dict())
        z=torch.randn(7,8,dtype=torch.double,requires_grad=True)
        mat=torch.zeros(7,7,dtype=torch.double)
        mat[a.edge[0],a.edge[1]]=a.weight
        h=a.norm(z)
        expected=z+.1*a.output(F.gelu(a.input(.5*h+.5*mat@h)))
        actual=a(z)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        da=torch.autograd.grad(actual.square().sum(),z,retain_graph=True)[0]
        de=torch.autograd.grad(expected.square().sum(),z)[0]
        torch.testing.assert_close(da,de,atol=1e-11,rtol=1e-11)
        j=int(a.edge[1,a.edge[0]==0][0]); altered=z.detach().clone(); altered[j,0]+=4
        torch.testing.assert_close(b(altered)[0],b(z)[0],atol=0,rtol=0)
        self.assertGreater(float((a(altered)[0]-a(z)[0]).abs().max()),1e-7)

    def test_common_initialization_rng_and_zero_residual_equivalence(self):
        torch.manual_seed(0); base=RelationSheafModel(self.dd,self.cfg,self.graphs,'content',training_seed=0)
        rng=torch.get_rng_state(); models=[]
        for mode in ('self','population'):
            torch.manual_seed(0); m=LateReadoutModel(self.dd,self.cfg,self.graphs,mode,training_seed=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            params=dict(m.named_parameters())
            for name,value in base.named_parameters():
                target=name.replace('classifier.','classifier.1.',1) if name.startswith('classifier.') else name
                torch.testing.assert_close(value,params[target],atol=0,rtol=0)
            m.classifier[0].scale=0
            for training in (False,True):
                base.train(training); m.train(training)
                torch.manual_seed(33); ref=base(self.x)[0]
                torch.manual_seed(33); actual=m(self.x)[0]
                torch.testing.assert_close(ref,actual,atol=0,rtol=0)
            m.classifier[0].scale=.1; models.append(m)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))
        for (na,a),(nb,b) in zip(models[0].named_parameters(),models[1].named_parameters()):
            self.assertEqual(na,nb); torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_all_added_parameters_receive_gradients(self):
        for mode in ('self','population'):
            torch.manual_seed(17); layer=LatePatientReadout(8,7,self.graphs,mode)
            z=torch.randn(7,8); probe=torch.randn(7,8)
            (layer(z)*probe).sum().backward()
            for name,p in layer.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                self.assertGreater(float(p.grad.abs().sum()),0,name)


if __name__=='__main__': unittest.main()
