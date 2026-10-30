import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_contracts import sample


class ReviewFixes(unittest.TestCase):
    def test_production_minimum_scene_support_cannot_be_weakened(self):
        from proc_research.calibration.run import ROOT,load_config
        cfg=json.loads((ROOT/'research/p01/config.json').read_text());cfg['calibration']['min_scenes']=1
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_config(p)
    def test_confirmation_claim_cannot_change_with_seal_filename(self):
        from proc_research.calibration.run import claim_confirmation
        with tempfile.TemporaryDirectory() as d:
            ident={'manifest_sha256':'a'*64,'config_sha256':'b'*64}
            claim_confirmation(Path(d),ident,'run1','seal1')
            claim_confirmation(Path(d),ident,'run1','seal1') # exact resume permitted
            with self.assertRaises(ValueError):claim_confirmation(Path(d),ident,'run2','seal2')
    def test_code_identity_tracks_untracked_runtime_shell(self):
        from proc_research.calibration.run import code_files
        self.assertIn('scripts/p01/python.sh',code_files())
        self.assertIn('research/p01/config.json',code_files())
        self.assertNotIn('research/p01/run_manifest.json',code_files())
    def test_receipt_failure_not_silently_accepted(self):
        from proc_research.calibration.run import _receipt
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'complete.json').write_text(json.dumps({'status':'FAIL','files':{}}))
            (p/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):_receipt(p)
    def test_background_and_agreement_not_replaced_by_argmax(self):
        from proc_research.calibration.signatures import fit_signatures,CalibrationConfig
        from proc_research.calibration.decode import decode_multiview
        v=[sample(scores=torch.tensor([.01,.7,.6])[:,None,None].expand(3,8,8).clone()),
           sample(view=504,scores=torch.tensor([.01,.7,.6])[:,None,None].expand(3,8,8).clone())]
        bank=fit_signatures(v,CalibrationConfig(min_pixels=1))
        ra=np.array([[.1,.1,.1],[.01,.8,.8],[.01,.7,.6]],dtype='float32')
        sig=replace(bank.global_bank,R_a=ra,R_e=ra.copy(),log_mass=np.log(np.full(3,1/3,dtype='float32')),
                    mass_argmax=np.full(3,5000),m_bg=.01)
        bank.global_bank=sig
        y,s=decode_multiview(v,bank)
        self.assertTrue((y==2).all());self.assertEqual(s['fallback_pixels'],0)
        re=ra.copy();re[[1,2]]=re[[2,1]];bank.global_bank=replace(sig,R_e=re)
        y,s=decode_multiview(v,bank)
        self.assertTrue((y==1).all());self.assertEqual(s['disagreement_pixels'],64)
        # A high null prior must compete, rather than being discarded.
        bank.global_bank=replace(sig,m_bg=1.);bank.config=replace(bank.config,background_offset=100.)
        y,s=decode_multiview(v,bank)
        self.assertTrue((y==0).all());self.assertEqual(s['fallback_pixels'],0)
    def test_row_fallback_counts_all_views_and_signature_winners(self):
        from proc_research.calibration.decode import signature_fallback_mask
        winners=[torch.tensor([0,0]),torch.tensor([1,0])]
        rows=[torch.tensor([False,True,False]),torch.tensor([True,False,False])]
        self.assertEqual(signature_fallback_mask(rows,winners).tolist(),[True,True])
    def test_differing_active_vocabularies_rejected(self):
        from proc_research.calibration.signatures import fit_signatures
        from proc_research.calibration.decode import decode_multiview
        v=[sample(),sample(view=504)];bank=fit_signatures(v)
        v[1].active=torch.tensor([0,1])
        with self.assertRaises(ValueError):decode_multiview(v,bank)


if __name__=='__main__':unittest.main()
