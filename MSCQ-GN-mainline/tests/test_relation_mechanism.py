"""Protect paired experimental summaries from partial folds or patient mixups."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relation_mechanism import aggregate,folder_for
from relation_experiment import dump


class MechanismSummaryTests(unittest.TestCase):
    def write_result(self,out,mode,fold,predictions,offset=0):
        folder=folder_for(out,0,mode,fold)
        folder.mkdir(parents=True,exist_ok=True)
        labels=np.array([0,1,0,1]);predictions=np.array(predictions)
        acc=float((predictions==labels).mean())
        dump(folder/'complete.json',dict(seed=0,fold=fold,mode=mode,
             best=dict(acc=acc,macro_f1=acc,auc=.5),last=dict(acc=acc)))
        np.savez_compressed(folder/'predictions.npz',labels=labels,
             logits=np.eye(2)[predictions],row_index=np.arange(4)+10*fold+offset)

    def test_partial_folds_not_reported_as_complete_and_all_folds_get_equal_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            self.write_result(out,'content',1,[0,0,0,0])
            report=aggregate(out,[0],[1,2],True)
            self.assertEqual(report['paired'],[])
            self.write_result(out,'static',1,[0,1,0,0])
            report=aggregate(out,[0],[1,2],True)
            self.assertEqual(report['complete_paired_seeds'],[])
            self.assertIsNone(report['mean_delta_complete_paired_seeds'])
            self.assertEqual(report['paired'][0]['mean_delta'],.25)
            self.write_result(out,'content',2,[0,1,0,0])
            self.write_result(out,'static',2,[0,0,0,0])
            report=aggregate(out,[0],[1,2],True)
            self.assertEqual(report['complete_paired_seeds'],[0])
            self.assertEqual(report['mean_delta_complete_paired_seeds'],0.)
            self.assertEqual(report['paired'][0]['mean_delta_excluding_pilot_fold'],-.25)
            self.assertEqual((report['paired'][0]['wins'],report['paired'][0]['losses']),(1,1))
            self.assertEqual((report['paired'][0]['corrected'],report['paired'][0]['harmed']),(1,1))

    def test_rejects_pairing_predictions_for_different_patients(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            self.write_result(out,'content',1,[0,0,0,0])
            self.write_result(out,'static',1,[0,1,0,0],offset=1000)
            with self.assertRaisesRegex(RuntimeError,'different patients'):
                aggregate(out,[0],[1],True)


if __name__=='__main__':
    unittest.main(verbosity=2)
