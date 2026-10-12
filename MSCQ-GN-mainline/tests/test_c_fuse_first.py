import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel
from Model.c_fuse_first import FuseFirstModel


class FuseFirstTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x,self.graphs,self.dd,self.cfg=tiny_case()

    def test_all_parameters_and_initialization_match_c(self):
        torch.manual_seed(19)
        c=RelationSheafModel(self.dd,self.cfg,self.graphs,'content',training_seed=0)
        state=torch.get_rng_state()
        torch.manual_seed(19)
        d=FuseFirstModel(self.dd,self.cfg,self.graphs,training_seed=0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(list(c.state_dict()),list(d.state_dict()))
        for k,v in c.state_dict().items():torch.testing.assert_close(v,d.state_dict()[k],atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in c.parameters()),sum(p.numel() for p in d.parameters()))

    def test_actual_graph_inputs_are_identical_fused_vectors_and_after_readout(self):
        model=FuseFirstModel(self.dd,self.cfg,self.graphs).eval()
        trace={};order=[];handles=[]
        def readout_hook(module,args,out):
            trace['fused']=out.detach().clone();order.append('fusion')
        def prop_hook(module,args):
            trace['graph_input']=args[0].detach().clone();order.append('graph')
        handles.append(model.Message_MLP.register_forward_hook(readout_hook))
        handles.append(model.propagation.register_forward_pre_hook(prop_hook))
        try:
            with torch.no_grad():out=model(self.x)
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(order,['fusion','graph'])
        for n in range(2):torch.testing.assert_close(trace['graph_input'][:,n],trace['fused'],atol=0,rtol=0)
        self.assertEqual(out[0].shape,(7,3))

    def test_no_graph_bypass_and_auxiliary_location(self):
        model=FuseFirstModel(self.dd,self.cfg,self.graphs).eval()
        original=model(self.x)
        handle=model.propagation.register_forward_hook(lambda module,args,out:out+2.0)
        try:changed=model(self.x)
        finally:handle.remove()
        self.assertGreater(float((changed[0]-original[0]).abs().max()),1e-6)
        torch.testing.assert_close(original[1:],changed[1:],atol=0,rtol=0)
        changed[0].square().sum().backward()
        for name,p in model.named_parameters():
            if name.startswith('propagation.'):
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__=='__main__':unittest.main()
