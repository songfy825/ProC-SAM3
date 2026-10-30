import json
from pathlib import Path
import tempfile
import unittest
from proc_research.cache import write_json,sha256
from proc_research.protocol import create_run


class StageGates(unittest.TestCase):
    def test_seal_bound_bank_required_in_confirmation_decode(self):
        from proc_research.calibration.run import validate_bank_binding
        ident={'stage':'confirmation','frozen_bank_receipt_sha256':'a'*64}
        with self.assertRaises(ValueError):validate_bank_binding(ident,'b'*64)
        validate_bank_binding(ident,'a'*64)
    def test_resume_g0_requires_saved_benchmark(self):
        from proc_research.calibration.run import g0_dataset_complete
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);row={'image_id':'x'}
            have={('LoveDA','x',v) for v in (1008,504)}
            self.assertFalse(g0_dataset_complete(p,'LoveDA',[row],have))
            write_json(p/'online_benchmark_LoveDA.json',{'warmup_pairs':5,'timed_pairs':3})
            self.assertTrue(g0_dataset_complete(p,'LoveDA',[row],have))
    def test_budget_selection_can_reduce_with_evidence(self):
        from proc_research.protocol import bounded_limits
        self.assertEqual(bounded_limits({'calibration':10,'pilot':10,'confirmation':30})['confirmation'],30)
        with self.assertRaises(ValueError):bounded_limits({'calibration':51,'pilot':50,'confirmation':200})
    def test_capacity_plan_includes_all_splits(self):
        from proc_research.prepare import capacity_plan
        rows=[{'dataset':'LoveDA','height':100,'width':200,'split':s} for s in ('calibration','pilot','confirmation')]
        r=capacity_plan(rows,{'LoveDA':(10,7)})
        self.assertEqual(r['total_cache_bytes'],sum(r['by_split'].values()))
        self.assertGreater(r['total_cache_bytes'],3*100*200*2*4*17)
    def test_resume_capacity_only_accounts_for_missing_views(self):
        from proc_research.calibration.run import remaining_cache_bytes
        rows=[{'dataset':'LoveDA','image_id':'x','height':100,'width':200}]
        all_bytes=remaining_cache_bytes(rows,set(),10,7)
        half=remaining_cache_bytes(rows,{('LoveDA','x',1008)},10,7)
        self.assertEqual(all_bytes,2*half)
        self.assertEqual(remaining_cache_bytes(rows,{('LoveDA','x',1008),('LoveDA','x',504)},10,7),0)
    def test_reordered_manifest_cannot_reuse_confirmation_images(self):
        from proc_research.calibration.run import claim_confirmation
        with tempfile.TemporaryDirectory() as d:
            a={'manifest_sha256':'a'*64,'config_sha256':'b'*64,'confirmation_images':['c'*64,'d'*64]}
            b=dict(a,manifest_sha256='e'*64,confirmation_images=['d'*64,'c'*64])
            claim_confirmation(Path(d),a,'run1','seal1')
            with self.assertRaises(ValueError):claim_confirmation(Path(d),b,'run2','seal2')


if __name__=='__main__':unittest.main()
