import sys
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Model.c_mechanism_controls import MechanismPropagation, scalar_diffusion
from Model.relation_sheaf import RelationContentPropagation, modal_candidates, cooperative_operator, conformal_maps


class Controls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.graphs=modal_candidates(torch.randn(7,6),[[0,1,2],[3,4,5]],2)
        self.base=RelationContentPropagation(8,2,7,self.graphs,'content').double()
        self.x=torch.randn(7,2,8,dtype=torch.double)

    def test_scalar_matches_dense_and_identity_sheaf_gradients(self):
        edge=self.graphs[0]['edge']
        x=torch.randn(7,8,dtype=torch.double,requires_grad=True)
        w=torch.rand(edge.shape[1],dtype=torch.double,requires_grad=True)
        a=torch.zeros(7,7,dtype=torch.double).index_put(tuple(edge),w)
        d=a.sum(1);inv=(d+1).rsqrt()
        lap=inv[:,None]*(torch.diag(d)-a)*inv[None,:]
        actual=scalar_diffusion(x,edge,w)
        dense=lap@lap@x
        torch.testing.assert_close(actual,dense,atol=1e-12,rtol=1e-12)
        maps=conformal_maps(torch.zeros(7,4,dtype=torch.double))
        identity=cooperative_operator(x.reshape(7,2,4),edge,w,*maps).flatten(1)
        torch.testing.assert_close(actual,identity,atol=1e-12,rtol=1e-12)
        probe=torch.randn_like(x)
        ga=torch.autograd.grad((actual*probe).sum(),(x,w),retain_graph=True)
        gb=torch.autograd.grad((dense*probe).sum(),(x,w))
        for left,right in zip(ga,gb):torch.testing.assert_close(left,right,atol=1e-11,rtol=1e-11)

    def test_d1_matches_explicit_one_graph_per_content(self):
        p=MechanismPropagation.from_original(self.base,self.graphs,'D1').double()
        z=self.base.input_norm(self.x)
        q=F.normalize(self.base.edge_query(z),dim=-1)
        v=self.base.value(z)
        pieces=[]
        for n,g in enumerate(self.graphs):
            i,j=g['edge']
            weight=torch.sigmoid((q[i,n]*q[j,n]).sum(-1)+g['similarity']*
                self.base.edge_similarity[n,n]+self.base.edge_bias[n,n])
            params=self.base.conditioner(self.base.content_id(torch.tensor(n))).expand(7,-1)
            pieces.append(cooperative_operator(v[:,n].reshape(7,2,4),g['edge'],weight,
                *conformal_maps(params)).reshape(7,8))
        delta=torch.stack(pieces,1)
        expected=self.base.output_norm(self.x-self.base.step_logit.sigmoid()*self.base.output(F.gelu(delta)))
        torch.testing.assert_close(p(self.x),expected,atol=1e-7,rtol=1e-7)
        # Nonmatching graph content never appears; exactly sum of six graph supports.
        self.assertEqual(p.edge.shape[1],sum(g['edge'].shape[1] for g in self.graphs))
        self.assertEqual(p.edge_similarity.numel(),2)

    def test_c_is_exact_and_d2_is_identity_map_model(self):
        c=MechanismPropagation.from_original(self.base,self.graphs,'C')
        torch.testing.assert_close(c(self.x),self.base(self.x),atol=0,rtol=0)
        d2=MechanismPropagation.from_original(self.base,self.graphs,'D2')
        with torch.no_grad():
            self.base.conditioner[-1].weight.zero_();self.base.conditioner[-1].bias.zero_()
        torch.testing.assert_close(d2(self.x),self.base(self.x),atol=1e-12,rtol=1e-12)
        self.assertFalse(any('conditioner' in n or 'content_id' in n for n,_ in d2.named_parameters()))

    def test_all_retained_parameters_participate(self):
        for mode in ('D1','D2'):
            p=MechanismPropagation.from_original(self.base,self.graphs,mode).double()
            out=p(self.x);(out*torch.randn_like(out)).sum().backward()
            for name,v in p.named_parameters():
                self.assertIsNotNone(v.grad,name)
                self.assertTrue(torch.isfinite(v.grad).all(),name)
                self.assertGreater(float(v.grad.abs().max()),0,name)


if __name__=='__main__':unittest.main()
