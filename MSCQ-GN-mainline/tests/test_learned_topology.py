import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import cooperative_operator,conformal_maps,RelationSheafModel
from Model.learned_topology import (_SoftBudget,select_neighbors,dense_cooperative_operator,LearnedTopologyModel)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_operator_matches_original_values_and_gradients(self):
        groups,n=2,5
        x=torch.randn(groups,n,2,3,dtype=torch.double,requires_grad=True)
        p=(torch.randn(groups,n,4,dtype=torch.double)*.1).requires_grad_()
        a=torch.rand(groups,n,n,dtype=torch.double,requires_grad=True)
        adjacency=(a+a.transpose(-1,-2))/2*(1-torch.eye(n,dtype=torch.double))
        edge=torch.cat([adjacency[g].nonzero().T+g*n for g in range(groups)],1)
        weight=torch.cat([adjacency[g][tuple(adjacency[g].nonzero().T)] for g in range(groups)])
        dense=dense_cooperative_operator(x,adjacency,*conformal_maps(p))
        sparse=cooperative_operator(x.reshape(groups*n,2,3),edge,weight,
            *conformal_maps(p.reshape(groups*n,4))).reshape_as(dense)
        torch.testing.assert_close(dense,sparse,atol=1e-11,rtol=1e-11)
        probe=torch.randn_like(dense)
        ga=torch.autograd.grad((dense*probe).sum(),(x,p,a),retain_graph=True)
        gb=torch.autograd.grad((sparse*probe).sum(),(x,p,a))
        for left,right in zip(ga,gb):
            torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)

    def test_soft_budget_constraint_and_implicit_gradient(self):
        score=torch.randn(2,5,5,dtype=torch.double,requires_grad=True)*.2
        valid=~torch.eye(5,dtype=torch.bool)
        p=_SoftBudget.apply(score,valid,2,.2)
        torch.testing.assert_close(p.sum(-1),torch.full((2,5),2.,dtype=torch.double),atol=1e-11,rtol=0)
        self.assertFalse(bool(p.diagonal(dim1=-2,dim2=-1).any()))
        self.assertTrue(torch.autograd.gradcheck(lambda s:_SoftBudget.apply(s,valid,2,.2),(score,),
                                                eps=1e-5,atol=2e-5,rtol=2e-4))

    def test_hard_forward_budget_and_absent_edge_surrogate_gradient(self):
        score=torch.randn(2,7,7,requires_grad=True)
        mask,hard,neighbors=select_neighbors(score,2,.2)
        torch.testing.assert_close(mask,hard.float(),rtol=0,atol=0)
        self.assertEqual(neighbors.shape,(2,7,2))
        self.assertTrue(torch.equal(hard,hard.transpose(-1,-2)))
        self.assertFalse(bool(hard.diagonal(dim1=-2,dim2=-1).any()))
        (mask*torch.randn_like(mask)).sum().backward()
        absent=(~hard)&(~torch.eye(7,dtype=torch.bool))
        self.assertGreater(float(score.grad[absent].abs().sum()),0.)
        torch.testing.assert_close(score.grad.sum(-1),torch.zeros(2,7),atol=1e-5,rtol=0)
        _,changed,_=select_neighbors(-score.detach(),2,.2)
        self.assertTrue(bool((changed&~hard).any()))

    def test_initial_models_and_rng_match_original_c(self):
        x,graphs,dd,cfg=tiny_case()
        models=[];states=[]
        for mode in ('fixed','learned','original'):
            torch.manual_seed(9)
            model=(RelationSheafModel(dd,cfg,graphs,'content',training_seed=9) if mode=='original'
                   else LearnedTopologyModel(dd,cfg,graphs,x,9,mode,k=2))
            models.append(model.eval());states.append(torch.get_rng_state())
        self.assertTrue(all(torch.equal(states[0],s) for s in states))
        for model in models[:2]:
            for name,value in models[2].state_dict().items():
                torch.testing.assert_close(model.state_dict()[name],value,atol=0,rtol=0)
            torch.testing.assert_close(model(x)[0],models[2](x)[0],atol=2e-6,rtol=2e-6)
            self.assertTrue(torch.equal(model.propagation.last_support,model.propagation.fixed_support))
        torch.testing.assert_close(models[0](x)[0],models[1](x)[0],atol=0,rtol=0)
        for model in models[:2]:
            model.train()
            logits=model(x)[0]
            F.cross_entropy(logits,torch.arange(7)%3).backward()
            self.assertGreater(float(model.propagation.relation_blend.grad.abs().sum()),0.)
            self.assertGreater(float(model.propagation.edge_query.weight.grad.abs().sum()),0.)

    def test_eval_checkpoint_roundtrip_and_full_cohort_order(self):
        x,graphs,dd,cfg=tiny_case()
        first=LearnedTopologyModel(dd,cfg,graphs,x,0,'learned',k=2).eval()
        with torch.no_grad():first.propagation.relation_blend.fill_(.7)
        second=LearnedTopologyModel(dd,cfg,graphs,x,0,'learned',k=2).eval()
        second.load_state_dict(first.state_dict())
        torch.testing.assert_close(first(x)[0],second(x)[0],atol=0,rtol=0)
        self.assertTrue(torch.equal(first.propagation.last_neighbors,second.propagation.last_neighbors))
        with self.assertRaises(ValueError):first(x[:-1])


if __name__=='__main__':unittest.main()
