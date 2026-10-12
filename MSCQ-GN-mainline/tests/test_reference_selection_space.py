import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_selection_space import best_masks,analyze_arrays
from Utils.reference_selection import coalition_masks


class SelectionSpaceTests(unittest.TestCase):
    def test_numerical_ties_keep_all_open(self):
        masks=coalition_masks(2)
        loss=np.ones((4,3))
        loss[0,0]-=2e-8
        np.testing.assert_array_equal(best_masks(loss,masks),[3,3,3])
        loss[0,1]-=1e-3
        self.assertEqual(best_masks(loss,masks)[1],0)

    def test_nll_oracle_and_acc_oracle_are_distinguished(self):
        # Highest true-class probability need not maximize classification ACC.
        probs=np.array([[[.45,.50,.05],[.60,.25,.15]],
                        [[.40,.30,.30],[.10,.70,.20]]])
        logits=np.log(probs); y=np.array([0,0]); losses=-np.log(probs[:,:,0])
        result,choices=analyze_arrays(logits,losses,y,coalition_masks(1))
        np.testing.assert_array_equal(choices,[0,0])
        self.assertEqual(result['individual_nll_oracle']['correct'],1)
        self.assertEqual(result['individual_acc_oracle']['correct'],2)
        self.assertEqual(result['individual_acc_oracle']['additional_correct_over_global_acc'],1)

    def test_no_selection_space_when_outputs_identical(self):
        probs=np.array([[.7,.3],[.4,.6],[.8,.2]])
        logits=np.repeat(np.log(probs)[None],4,axis=0)
        labels=np.array([0,1,1])
        losses=-np.log(np.repeat(probs[None],4,axis=0)[:,np.arange(3),labels])
        result,choices=analyze_arrays(logits,losses,labels,coalition_masks(2))
        np.testing.assert_array_equal(choices,[3,3,3])
        self.assertEqual(result['prediction_changes']['invariant_count'],3)
        self.assertEqual(result['individual_acc_oracle']['additional_correct_over_global_acc'],0)


if __name__=='__main__': unittest.main()
