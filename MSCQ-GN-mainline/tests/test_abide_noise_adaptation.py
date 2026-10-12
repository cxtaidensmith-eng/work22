import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_relation_sheaf import tiny_case
from abide_noise_adaptation import make_model


class NoiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x,self.graphs,self.dd,self.cfg=tiny_case()
        self.dd['Modal_Name']=['PHENO','Correlation'];self.cfg.input_noise_std=.05

    def test_parameters_eval_rng_and_exact_input_perturbation(self):
        targets={'W':[.15,.15],'N1':[.15,.03],'N2':[0.,.15],'N3':[0.,.03]}
        models={};states={}
        for arm in targets:
            torch.manual_seed(0)
            models[arm]=make_model(self.dd,self.cfg,self.graphs,'content',training_seed=0,arm=arm)
            states[arm]=torch.get_rng_state()
            torch.testing.assert_close(models[arm]._modal_noise_std*models[arm].noise_scale,torch.tensor(targets[arm]))
        ref=models['W']; ref.eval()
        with torch.no_grad():reference=ref(self.x)[0]
        for arm,model in models.items():
            self.assertTrue(torch.equal(states['W'],states[arm]))
            for (n,a),(m,b) in zip(ref.named_parameters(),model.named_parameters()):
                self.assertEqual(n,m);torch.testing.assert_close(a,b,atol=0,rtol=0)
            model.eval()
            with torch.no_grad():torch.testing.assert_close(model(self.x)[0],reference,atol=0,rtol=0)
            cache={}
            def capture(mod,args):cache['x']=args[0].detach().clone()
            handle=model.modal_token_encoder.register_forward_pre_hook(capture)
            model.train();torch.manual_seed(19)
            expected_noise=torch.randn_like(self.x)
            torch.manual_seed(19);model(self.x)
            after=torch.get_rng_state();handle.remove()
            expected=model.Feature_Modal(self.x).detach()+expected_noise*torch.tensor(targets[arm])[model.feature_to_modal]
            torch.testing.assert_close(cache['x'],expected,atol=1e-7,rtol=1e-7)
            if arm=='W':training_rng=after
            else:self.assertTrue(torch.equal(training_rng,after))
            clone=make_model(self.dd,self.cfg,self.graphs,'content',training_seed=0,arm='W')
            clone.load_state_dict(model.state_dict())
            torch.testing.assert_close(clone._modal_noise_std,model._modal_noise_std,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
