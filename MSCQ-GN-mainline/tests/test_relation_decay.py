"""Check the decay intervention and that grouping alone preserves Adam updates."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from test_relation_sheaf import tiny_case
from Model.relation_sheaf import RelationSheafModel
from Utils import CustomCosineAnnealingLR
from Utils.graph_diagnostics import GraphDiagnostics
from Utils.relation_optimizer import relation_optimizer, optimizer_manifest


class GraphDecayTests(unittest.TestCase):
    def model_and_config(self, device='cpu'):
        _, graphs, dd, cfg = tiny_case()
        cfg.lr, cfg.weight_decay = .006, .0007
        torch.manual_seed(7)
        return RelationSheafModel(dd,cfg,graphs,'static').to(device),cfg

    def test_only_graph_parameters_lose_decay_and_observer_uses_effective_groups(self):
        model,cfg = self.model_and_config()
        optimizer = relation_optimizer(model,cfg,0.)
        manifest = optimizer_manifest(model,optimizer)
        names = [name for group in manifest for name in group['names']]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(set(names),set(dict(model.named_parameters())))
        graph_names = set(manifest[1]['names'])
        self.assertIn('propagation.value.weight',graph_names)
        self.assertIn('propagation.conditioner.2.weight',graph_names)
        self.assertIn('propagation.input_norm.weight',graph_names)
        self.assertNotIn('propagation.output_norm.weight',graph_names)
        before = {name:p.detach().clone() for name,p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        with tempfile.TemporaryDirectory() as directory:
            observer = GraphDiagnostics(model,Path(directory)/'trace.csv',cfg.weight_decay,optimizer=optimizer)
            try:
                observer.begin(1,1,cfg.lr)
                observer.after_backward()
                observer.before_step(0.)
                for group in ('value','output','maps','edges','step','graph_input_norm'):
                    self.assertEqual(observer.row[group+'_decay_grad'],0.)
                self.assertGreater(observer.row['patient_path_decay_grad'],0.)
                optimizer.step()
                for name in graph_names:
                    torch.testing.assert_close(dict(model.named_parameters())[name],before[name],rtol=0,atol=0)
                for name in ('propagation.output_norm.weight','classifier.0.weight'):
                    self.assertFalse(torch.equal(dict(model.named_parameters())[name],before[name]))
            finally:
                observer.close()

    def grouped_adam_matches(self,device):
        original,cfg = self.model_and_config(device)
        models = [original,copy.deepcopy(original),copy.deepcopy(original)]
        optimizers = [torch.optim.Adam(models[0].parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay),
                      relation_optimizer(models[1],cfg),relation_optimizer(models[2],cfg,cfg.weight_decay)]
        schedules = [CustomCosineAnnealingLR(opt,T_max=1000,eta_min=.0001) for opt in optimizers]
        for step in range(25):
            for model,opt,schedule in zip(models,optimizers,schedules):
                for i,p in enumerate(model.parameters()):
                    p.grad = torch.full_like(p,(i+1)*.001+step*.0001)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();schedule.step()
            for model in models[1:]:
                for name,value in original.state_dict().items():
                    torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
            expected = optimizers[0].param_groups[0]['lr']
            for optimizer in optimizers:
                self.assertTrue(all(group['lr']==expected for group in optimizer.param_groups))

    def test_grouping_and_default_optimizer_preserve_cpu_updates_and_scheduler(self):
        self.grouped_adam_matches('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_grouping_and_default_optimizer_preserve_cuda_updates_and_scheduler(self):
        self.grouped_adam_matches('cuda')


if __name__=='__main__':
    unittest.main(verbosity=2)
