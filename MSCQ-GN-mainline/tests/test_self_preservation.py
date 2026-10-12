import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel
from self_preservation_pilot import (make_arm,patient_hash,new_optimizer,named_optimizer_state,
    restore_optimizer,bypass_logits,capture_rng,restore_rng)


class SelfPreservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.x,self.graphs,self.dd,self.cfg=tiny_case()
        self.cfg.Device=torch.device('cpu')
        self.cfg.lr,self.cfg.weight_decay=.006,.0007
        self.cfg.Use_scheduler,self.cfg.T_max,self.cfg.Lr_Min=True,1000,.0001
        self.base=RelationSheafModel(self.dd,self.cfg,self.graphs,'self_only',training_seed=0)

    @torch.no_grad()
    def test_graph_insertion_preserves_warm_predictions_and_rng_alignment(self):
        self.base.eval()
        reference=self.base(self.x)[0]
        arms=[make_arm(self.dd,self.cfg,self.graphs,0,a,self.base.state_dict()) for a in ('frozen_self','joint')]
        for model in arms:
            model.eval()
            torch.testing.assert_close(model(self.x)[0],reference,atol=1e-6,rtol=1e-6)
            self.assertEqual(patient_hash(model),patient_hash(self.base))
        state=capture_rng()
        for model in arms: model.train()
        a=arms[0](self.x)[0]
        restore_rng(state)
        b=arms[1](self.x)[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_continuation_preserves_adam_moments_and_scheduler(self):
        optimizer,scheduler=new_optimizer(self.base,self.cfg)
        for epoch in range(25):
            for i,p in enumerate(self.base.parameters()): p.grad=torch.full_like(p,(i+1)*.001)
            optimizer.step();scheduler.step()
        clone=copy.deepcopy(self.base)
        opt2,sch2=restore_optimizer(clone,self.cfg,named_optimizer_state(self.base,optimizer),
            scheduler.state_dict(),optimizer.param_groups[0]['lr'])
        for epoch in range(5):
            for model,opt,sch in ((self.base,optimizer,scheduler),(clone,opt2,sch2)):
                for i,p in enumerate(model.parameters()): p.grad=torch.full_like(p,(i+1)*.002+epoch*.01)
                opt.step();sch.step()
            self.assertEqual(scheduler.get_last_lr(),sch2.get_last_lr())
            for name,value in self.base.state_dict().items():
                torch.testing.assert_close(value,clone.state_dict()[name],atol=0,rtol=0)
        graph=make_arm(self.dd,self.cfg,self.graphs,0,'joint',self.base.state_dict())
        go,gs=restore_optimizer(graph,self.cfg,named_optimizer_state(self.base,optimizer),scheduler.state_dict(),optimizer.param_groups[0]['lr'])
        for p in graph.parameters(): p.grad=torch.zeros_like(p)
        for p in self.base.parameters(): p.grad=torch.zeros_like(p)
        go.step();gs.step();optimizer.step();scheduler.step()
        self.assertEqual(gs.get_last_lr(),scheduler.get_last_lr()*2)

    def test_frozen_patient_weights_remain_exact_and_graph_can_learn(self):
        model=make_arm(self.dd,self.cfg,self.graphs,0,'frozen_self',self.base.state_dict())
        before=patient_hash(model)
        opt,_=new_optimizer(model,self.cfg)
        original=model.propagation.output.weight.detach().clone()
        for _ in range(3):
            model.train();opt.zero_grad(set_to_none=True)
            model(self.x)[0].square().mean().backward();opt.step()
        self.assertEqual(patient_hash(model),before)
        self.assertFalse(torch.equal(original,model.propagation.output.weight))
        model.eval();self.base.eval()
        # Frozen parameters are bitwise equal (hash above); autograd-disabled
        # matrix kernels can differ by floating-point roundoff in their outputs.
        torch.testing.assert_close(bypass_logits(model,self.x),self.base(self.x)[0],atol=1e-7,rtol=1e-6)


if __name__=='__main__': unittest.main()
