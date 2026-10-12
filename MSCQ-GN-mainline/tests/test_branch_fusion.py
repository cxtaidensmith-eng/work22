"""Meaningful fusion equivalence, capacity activation, RNG and replay checks."""
import copy
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.relation_sheaf import RelationSheafModel, modal_candidates
from Model.branch_fusion import BranchFusionModel


def rng():
    return [torch.get_rng_state(),*torch.cuda.get_rng_state_all()]


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(91)
        self.x,self.dd,self.cfg=tiny_case()
        for k,v in dict(global_word_emb=None,semantic_branch='both',adj_mode='learned',
                        modal_structure='modal',readout_mode='class_query').items():
            setattr(self.cfg,k,v)
        self.graphs=modal_candidates(self.x,self.dd['Modal_Index'],k=2)

    def check(self,device):
        models={};streams={}
        for mode in ('old','C','concat','wide'):
            torch.manual_seed(18)
            models[mode]=(RelationSheafModel(self.dd,self.cfg,self.graphs,'content',training_seed=0)
                          if mode=='old' else BranchFusionModel(self.dd,self.cfg,self.graphs,mode)).to(device)
            streams[mode]=rng()
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(streams['old'],streams[mode])))
            for name,p in models['old'].state_dict().items():
                if mode in ('old','C') or not name.startswith('classifier.'):
                    torch.testing.assert_close(p,models[mode].state_dict()[name],atol=0,rtol=0)
        x=self.x.to(device)
        for training in (False,True):
            values={};streams={}
            for mode,model in models.items():
                torch.manual_seed(31);model.train(training)
                values[mode]=model(x);streams[mode]=rng()
                torch.testing.assert_close(values['old'],values[mode],atol=2e-6,rtol=2e-6)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(streams['old'],streams[mode])))
        # The copied concat head must retain backbone gradients initially.
        for mode in ('old','C','concat'):
            model=models[mode].eval();model.zero_grad(set_to_none=True)
            logits,emb,aux=model(x)
            loss=logits.square().sum()+sum(t.square().sum() for t in aux)
            loss.backward()
        for name,p in models['old'].named_parameters():
            for mode in ('C','concat'):
                q=dict(models[mode].named_parameters())[name]
                if mode=='concat' and name=='classifier.0.weight':
                    a,b=q.grad.chunk(2,-1)
                    torch.testing.assert_close(p.grad,a+b,atol=3e-5,rtol=3e-5)
                elif p.grad is not None:
                    torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-5)
                else:self.assertIsNone(q.grad)
        # Branch interventions zero only the final branch vector.
        for mode in ('C','concat','wide'):
            model=models[mode].eval();captured={}
            hooks=[model.Message_MLP.register_forward_hook(lambda m,a,o:captured.update(semantic=o)),
                   model.Global_Message.register_forward_hook(lambda m,a,o:captured.update(own=o))]
            try:
                with torch.no_grad():
                    baseline=model(x)[0]
                    for branch in ('semantic_only','own_only'):
                        actual=model(x,branch=branch)[0]
                        s,o=captured['semantic'],captured['own']
                        expected=model.classify(s if branch=='semantic_only' else torch.zeros_like(s),
                                                o if branch=='own_only' else torch.zeros_like(o))
                        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                    model.capture_diagnostics=True
                    before=rng();observed=model(x)[0]
                    torch.testing.assert_close(baseline,observed,atol=0,rtol=0)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,rng())))
            finally:
                for h in hooks:h.remove()
            replica=copy.deepcopy(model).eval()
            replica.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(model(x)[0],replica(x)[0],atol=0,rtol=0)
        # Additional units and independent concat weights really learn.
        for mode in ('concat','wide'):
            model=models[mode].eval()
            optimizer=torch.optim.Adam(model.classifier.parameters(),lr=.01)
            s=torch.randn(23,self.cfg.Hidden_size,device=device)
            o=torch.randn_like(s);labels=torch.arange(23,device=device)%self.dd['Class_Num']
            for _ in range(4):
                optimizer.zero_grad();F.cross_entropy(model.classify(s,o),labels).backward();optimizer.step()
            if mode=='concat':
                a,b=model.classifier[0].weight.chunk(2,-1)
                self.assertGreater(float((a-b).abs().max()),1e-5)
            else:
                width=self.cfg.Hidden_size//2
                self.assertGreater(float(model.classifier[3].weight[:,width:].abs().max()),1e-5)
                self.assertGreater(float(model.classifier[0].weight.grad[width:].abs().max()),1e-7)

    def test_cpu(self):self.check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):self.check('cuda')


if __name__=='__main__':unittest.main(verbosity=2)
