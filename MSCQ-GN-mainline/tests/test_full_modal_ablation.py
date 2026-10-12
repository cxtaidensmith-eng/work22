"""Checkpoint conversion and input removal remain valid after weight changes."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_modal_evidence import tiny_case
from Model.modal_evidence import ModalEvidenceModel
from Model.subset_fusion import SubsetFusionModel
from full_modal_ablation_experiment import load_full_state


class FullAblationTests(unittest.TestCase):
    def check_device(self,device):
        torch.set_num_threads(1)
        x,dd,cfg=tiny_case()
        x=x.to(device)
        source=ModalEvidenceModel(dd,cfg,'fusion',0).to(device)
        # Non-initial weights exercise the same conversion used on real checkpoints.
        with torch.no_grad():
            for p in source.parameters():
                p.add_(.013)
        for removed in [None]+dd['Modal_Name']:
            selected=[name for name in dd['Modal_Name'] if name!=removed]
            target=SubsetFusionModel(dd,cfg,selected,0).to(device)
            load_full_state(target,source.state_dict())
            indices=[dd['Modal_Name'].index(m) for m in selected]
            for training in (False,True):
                source.train(training);target.train(training)
                torch.manual_seed(801)
                full=source(x)
                after=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
                torch.manual_seed(801)
                actual=target(x)
                torch.testing.assert_close(actual['tokens'],full['tokens'][:,indices],atol=2e-6,rtol=2e-6)
                masked=torch.zeros_like(full['tokens']);masked[:,indices]=full['tokens'][:,indices]
                expected=source.fusion_head(masked.flatten(1))
                torch.testing.assert_close(actual['logits'],expected,atol=2e-6,rtol=2e-6)
                self.assertTrue(torch.equal(after[0],torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(after[1],torch.cuda.get_rng_state_all())))
                if removed is not None:
                    changed=x.clone();changed[:,dd['Modal_Index'][dd['Modal_Name'].index(removed)]]=999
                    torch.manual_seed(801)
                    torch.testing.assert_close(actual['logits'],target(changed)['logits'],atol=0,rtol=0)
        self.assertEqual(source.parameter_counts()['total'],
                         SubsetFusionModel(dd,cfg,dd['Modal_Name'],0).parameter_counts()['total'])

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        self.check_device('cuda')


if __name__=='__main__':
    unittest.main()
