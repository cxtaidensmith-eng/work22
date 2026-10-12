import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel
from Model.task_guided_graph import TaskGuidedGraphModel, balanced_relation_loss


class GuidedGraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_pair_loss_ignores_heldout_labels_edges_and_diagonal(self):
        score = torch.randn(2, 8, 8, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        fit = torch.arange(8) < 6
        loss = balanced_relation_loss(score, labels, fit)
        changed_labels = labels.clone(); changed_labels[~fit] = 99999
        changed_score = score.detach().clone()
        changed_score[:, ~fit, :] = 999
        changed_score[:, :, ~fit] = -999
        changed_score.diagonal(dim1=-2, dim2=-1).fill_(999)
        torch.testing.assert_close(loss, balanced_relation_loss(changed_score, changed_labels, fit), rtol=0, atol=0)
        loss.backward()
        self.assertEqual(float(score.grad[:, ~fit].abs().sum()), 0.)
        self.assertEqual(float(score.grad[:, :, ~fit].abs().sum()), 0.)
        self.assertEqual(float(score.grad.diagonal(dim1=-2,dim2=-1).abs().sum()), 0.)

    def test_class_group_and_positive_negative_balancing(self):
        y = torch.tensor([0,0,1,1,2,2,2])
        fit = torch.ones(7,dtype=torch.bool)
        score = torch.zeros(1,7,7,dtype=torch.double)
        for c,error in enumerate((1.,2.,3.)):
            mask=(y[:,None]==c)&(y[None,:]==c)
            score[:,mask]=1+error
        # Three positive class means have losses 1,4,9. Negatives all zero.
        torch.testing.assert_close(balanced_relation_loss(score,y,fit),torch.tensor(7/3,dtype=torch.double))

    def test_independent_query_supervision_and_topology_gradient(self):
        x,graphs,dd,cfg=tiny_case()
        model=TaskGuidedGraphModel(dd,cfg,graphs,x,0,k=2).train()
        with torch.no_grad(): model.propagation.relation_blend.fill_(.2)
        model(x)
        score=model.propagation.last_selection_score
        torch.testing.assert_close(score,score.transpose(-1,-2),atol=1e-7,rtol=1e-6)
        fit=torch.tensor([1,1,1,1,1,1,0],dtype=torch.bool)
        labels=torch.tensor([0,0,1,1,2,2,9999])
        balanced_relation_loss(score,labels,fit).backward()
        self.assertIsNone(model.propagation.edge_query.weight.grad)
        for p in (model.propagation.graph_query.weight,model.propagation.metric_weights,
                  model.propagation.relation_blend):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()),0.)
        self.assertNotEqual(model.propagation.graph_query.weight.data_ptr(),model.propagation.edge_query.weight.data_ptr())
        self.assertTrue(torch.equal(model.propagation.last_support,model.propagation.last_support.transpose(-1,-2)))

    def test_original_initialization_and_cpu_cuda_rng(self):
        x,graphs,dd,cfg=tiny_case()
        models=[];states=[]
        for guided in (False,True,True):
            torch.manual_seed(9)
            model=(TaskGuidedGraphModel(dd,cfg,graphs,x,9,k=2) if guided else
                   RelationSheafModel(dd,cfg,graphs,'content',training_seed=9)).eval()
            models.append(model)
            states.append([torch.get_rng_state(), *torch.cuda.get_rng_state_all()])
        for model,state in zip(models[1:],states[1:]):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(states[0],state)))
            for name,value in models[0].state_dict().items():
                torch.testing.assert_close(model.state_dict()[name],value,atol=0,rtol=0)
            torch.testing.assert_close(model(x)[0],models[0](x)[0],atol=2e-6,rtol=2e-6)
            self.assertTrue(torch.equal(model.propagation.last_support,model.propagation.fixed_support))
        for name,value in models[1].state_dict().items():
            torch.testing.assert_close(value,models[2].state_dict()[name],atol=0,rtol=0)

    def test_forced_original_support_ignores_metric_and_checkpoint_roundtrip(self):
        x,graphs,dd,cfg=tiny_case()
        model=TaskGuidedGraphModel(dd,cfg,graphs,x,0,k=2).eval()
        model.propagation.topology_mode='fixed'
        before=model(x)[0]
        with torch.no_grad():
            model.propagation.graph_query.weight.normal_()
            model.propagation.metric_weights.normal_()
            model.propagation.relation_blend.fill_(.5)
        torch.testing.assert_close(before,model(x)[0],atol=0,rtol=0)
        model.propagation.topology_mode='learned'
        clone=TaskGuidedGraphModel(dd,cfg,graphs,x,0,k=2).eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x)[0],clone(x)[0],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
