"""Frozen acceptance contracts; synthetic data never certifies model inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch


class Availability(unittest.TestCase):
    def test_p01_modules_exist(self):
        root = Path(__file__).resolve().parents[2]
        for path in ('cache.py', 'geometry.py', 'protocol.py', 'evaluate.py',
                     'statistics.py', 'baseline.py', 'calibration/decode.py',
                     'calibration/signatures.py', 'calibration/run.py'):
            self.assertTrue((root/'proc_research'/path).is_file(), path)


def sample(scene='s0', view=1008, image='i0', scores=None):
    from proc_research.cache import SampleCache
    if scores is None:
        scores = torch.tensor([.1,.9,.2], dtype=torch.float32)[:,None,None].expand(3,8,12).clone()
    return SampleCache(image, scene, 'LoveDA', view, scores, scores.clone(),
                       torch.tensor([.3,.9,.4]), torch.tensor([.3,.9,.4]),
                       torch.arange(3), torch.arange(3), (0,1,2),
                       scores.argmax(0), {'kind':'full_resize','original_hw':list(scores.shape[-2:]),
                       'encoder_input':[1,3,view,view], 'feature_shapes':[[1,8,view//14,view//14]]},
                       {'image_sha256':'a'*64,'weight_sha256':'b'*64,'pool_sha256':'c'*64,
                        'config_sha256':'d'*64,'code_sha256':'e'*64})


class Geometry(unittest.TestCase):
    def test_non_square_and_borders(self):
        from proc_research.geometry import align_scores
        x=torch.tensor([[[0.,1.],[2.,3.]]])
        y=align_scores(x,(3,5))
        self.assertEqual(tuple(y.shape),(1,3,5))
        self.assertEqual(y[0,0,0].item(),0)
        self.assertEqual(y[0,-1,-1].item(),3)
        torch.testing.assert_close(y[0,1],torch.tensor([1.,1.1,1.5,1.9,2.]))
    def test_identity_does_not_interpolate(self):
        from proc_research.geometry import align_scores
        x=torch.rand(3,7,11)
        self.assertIs(align_scores(x,(7,11)),x)
    def test_label_interpolation_rejected(self):
        from proc_research.geometry import align_scores
        with self.assertRaises(ValueError): align_scores(torch.zeros(1,2,2,dtype=torch.long),(4,4))
    def test_actual_encoder_guard(self):
        from proc_research.geometry import validate_encoder
        validate_encoder(504,[1,3,504,504])
        with self.assertRaises(ValueError): validate_encoder(504,[1,3,1008,1008])


class Cache(unittest.TestCase):
    def test_lossless_roundtrip_and_corruption(self):
        from proc_research.cache import write_sample,load_sample
        with tempfile.TemporaryDirectory() as tmp:
            a=sample(); receipt=write_sample(Path(tmp)/'one',a)
            b=load_sample(Path(tmp)/'one',expected=receipt)
            self.assertTrue(torch.equal(a.scores,b.scores));self.assertTrue(torch.equal(a.prediction,b.prediction))
            with open(Path(tmp)/'one'/'arrays.npz','ab') as f: f.write(b'corruption')
            with self.assertRaises(ValueError):load_sample(Path(tmp)/'one',expected=receipt)
    def test_metadata_identity_and_no_overwrite(self):
        from proc_research.cache import write_sample,load_sample
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'one'; receipt=write_sample(p,sample())
            with self.assertRaises(FileExistsError):write_sample(p,sample())
            meta=json.loads((p/'meta.json').read_text());meta['view']=504
            (p/'meta.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):load_sample(p,expected=receipt)
    def test_non_fp32_rejected(self):
        from proc_research.cache import validate_sample
        a=sample();a.scores=a.scores.half()
        with self.assertRaises(ValueError):validate_sample(a)
    def test_class_mapping_preserved(self):
        from proc_research.calibration.decode import fuse_mean
        a=sample();b=sample(view=504);b.class_ids=(1,0,2)
        with self.assertRaises(ValueError):fuse_mean([a,b])


class Decode(unittest.TestCase):
    def test_mean_continuous_not_labels(self):
        from proc_research.calibration.decode import fuse_mean,raw_labels
        a=sample(scores=torch.tensor([.1,.9,.0])[:,None,None]);b=sample(view=504,scores=torch.tensor([.8,.1,.0])[:,None,None])
        torch.testing.assert_close(fuse_mean([a,b]).flatten(),torch.tensor([.45,.5,0.]))
        self.assertEqual(raw_labels(fuse_mean([a,b]),.1,0).item(),1)
    def test_all_zero_is_background(self):
        from proc_research.calibration.decode import raw_labels
        self.assertTrue((raw_labels(torch.zeros(3,4,5),.1,2)==2).all())
    def test_rms_and_zero_factor(self):
        from proc_research.calibration.signatures import fit_rms
        from proc_research.calibration.decode import rms_labels
        a=sample(scores=torch.tensor([0.,.2,.8])[:,None,None].expand(3,8,8).clone())
        f=fit_rms([a]);torch.testing.assert_close(f,torch.tensor([1.,.2,.8]))
        # Scaling cannot turn a below-threshold winner into a detection.
        out=rms_labels([a],[torch.tensor([1.,.01,100.])],.3,0)
        self.assertTrue((out==0).all())
    def test_official_ecd_parity(self):
        from proc_research.calibration.decode import faithful_ecd
        from proc_research.vendor.ecd import ExclusiveConceptDecoder
        from proc_research.vendor.signatures import Signatures
        rng=np.random.default_rng(7)
        for bg in (None,0):
            for _ in range(10):
                c=4
                bank=Signatures(rng.uniform(.01,.99,(c,c)).astype('float32'),rng.uniform(.01,.99,(c,c)).astype('float32'),np.log(np.array([.1,.2,.3,.4],dtype='float32')),np.array([50,5000,5000,5000]),np.ones(c),.1,{'constants':{'CORE_MIN_PX':2000}})
                scores=torch.tensor(rng.random((c,4,7)),dtype=torch.float32);scores[:,0,0]=0
                presence=torch.tensor([.2,.7,.8,.4]);active=torch.tensor([0,2,3])
                expected=ExclusiveConceptDecoder(bank,.15,bg,device='cpu')(scores,presence,active)
                actual=faithful_ecd(scores,presence,active,bank,.15,bg)
                self.assertTrue(torch.equal(expected,actual))
    def test_ecd_independent_hand_calculation(self):
        from proc_research.calibration.decode import discriminants
        s=torch.tensor([[.8],[.2]]);r=torch.tensor([[.7,.3],[.1,.9]])
        d=discriminants(s,torch.zeros(2),r,.5)
        torch.testing.assert_close(d[:,0],torch.tensor([-.04,-1.96]))
    def test_multiview_disagreement_and_uncalibrated_fallback(self):
        from proc_research.calibration.signatures import fit_signatures,CalibrationConfig
        from proc_research.calibration.decode import decode_multiview,raw_labels,fuse_mean
        views=[sample(),sample(view=504)]
        bank=fit_signatures(views,CalibrationConfig(min_pixels=2000))
        labels,stats=decode_multiview(views,bank,conditional=True)
        self.assertTrue(torch.equal(labels,raw_labels(fuse_mean(views),.1,0)))
        self.assertEqual(stats['fallback_pixels'],96)
    def test_zero_evidence_does_not_nan(self):
        from proc_research.calibration.signatures import fit_signatures,CalibrationConfig
        from proc_research.calibration.decode import decode_multiview
        v=[sample(scores=torch.zeros(3,8,8)),sample(view=504,scores=torch.zeros(3,8,8))]
        b=fit_signatures(v,CalibrationConfig(min_pixels=1))
        labels,_=decode_multiview(v,b,conditional=True)
        self.assertTrue((labels==0).all())


class Signatures(unittest.TestCase):
    def test_scene_support_not_pixel_or_view_count(self):
        from proc_research.calibration.signatures import fit_signatures,CalibrationConfig
        data=[sample(scene='same',image=str(i),view=v) for i in range(6) for v in (504,1008)]
        bank=fit_signatures(data,CalibrationConfig(min_pixels=1))
        self.assertEqual(bank.support[504][1],1)
        self.assertTrue(bank.row_fallback[504][1])
        torch.testing.assert_close(torch.tensor(bank.conditioned[504].R_a),torch.tensor(bank.global_bank.R_a))
    def test_shared_priors_only_signature_changes(self):
        from proc_research.calibration.signatures import fit_signatures,CalibrationConfig
        data=[sample(scene=str(i),image=str(i),view=v) for i in range(6) for v in (504,1008)]
        b=fit_signatures(data,CalibrationConfig(min_pixels=1))
        for v in (504,1008):
            np.testing.assert_array_equal(b.conditioned[v].log_mass,b.global_bank.log_mass)
            np.testing.assert_array_equal(b.conditioned[v].uncalibrated,b.global_bank.uncalibrated)
            self.assertFalse(b.row_fallback[v][1])
    def test_shuffle_breaks_mapping_preserves_counts(self):
        from proc_research.calibration.signatures import shuffle_conditions
        data=[sample(scene=str(i),image=str(i),view=v) for i in range(10) for v in (504,1008)]
        for seed in (0,1,2):
            a=shuffle_conditions(data,seed)
            self.assertEqual(sorted(a),sorted(x.view for x in data))
            self.assertEqual(a,shuffle_conditions(data,seed))
            for view in (504,1008): self.assertEqual(len({a[i] for i,x in enumerate(data) if x.view==view}),2)


def row(i,split,scene=None):
    return dict(image_id=str(i),dataset='LoveDA',source_scene_id=scene or str(i),scene_evidence='verified source-map: fixture',split=split,image_path='/readonly/image.png',image_sha256=f'{i:064x}',height=8,width=12)


class Isolation(unittest.TestCase):
    def test_valid_and_scene_leak_negative(self):
        from proc_research.protocol import validate_manifest
        rows=[row(1,'calibration'),row(2,'pilot'),row(3,'confirmation')]
        validate_manifest(rows,verify_files=False)
        rows[-1]['source_scene_id']='1'
        with self.assertRaises(ValueError):validate_manifest(rows,verify_files=False)
    def test_image_leak_and_gt_rejected(self):
        from proc_research.protocol import validate_manifest
        rows=[row(1,'calibration'),row(2,'pilot')];rows[1]['image_sha256']=rows[0]['image_sha256']
        with self.assertRaises(ValueError):validate_manifest(rows,verify_files=False)
        a=row(1,'calibration');a['label_path']='secret'
        with self.assertRaises(ValueError):validate_manifest([a],verify_files=False)
    def test_scene_evidence_and_budget_required(self):
        from proc_research.protocol import validate_manifest
        a=row(1,'calibration');a['scene_evidence']=''
        with self.assertRaises(ValueError):validate_manifest([a],verify_files=False)
        with self.assertRaises(ValueError):validate_manifest([row(i,'calibration') for i in range(51)],verify_files=False)
    def test_run_identity_immutable(self):
        from proc_research.protocol import create_run
        with tempfile.TemporaryDirectory() as tmp:
            create_run(Path(tmp),'r1',{'sigma':.15})
            with self.assertRaises(FileExistsError):create_run(Path(tmp),'r1',{'sigma':.2})


class Evaluation(unittest.TestCase):
    def test_full_confusion_ignore_and_absent(self):
        from proc_research.evaluate import confusion_matrix,metrics
        cm=confusion_matrix(np.array([[0,1,255],[1,1,0]]),np.array([[0,0,2],[1,1,0]]),3,255)
        np.testing.assert_array_equal(cm,np.array([[2,0,0],[1,2,0],[0,0,0]]))
        result=metrics(cm);self.assertAlmostEqual(result['miou'],2/3);self.assertIsNone(result['iou'][2])
    def test_loveda_label_mapping(self):
        from proc_research.evaluate import map_labels
        np.testing.assert_array_equal(map_labels(np.array([0,1,7,255]),'LoveDA'),[255,0,6,255])
    def test_paired_scene_bootstrap(self):
        from proc_research.statistics import paired_bootstrap
        a=np.array([[[5,0],[0,5]],[[3,0],[0,7]]]);b=np.array([[[4,1],[1,4]],[[2,1],[1,6]]])
        r=paired_bootstrap(a,b,iterations=2000,seed=20261030)
        self.assertGreater(r['ci95_pp'][0],0);self.assertEqual(r,paired_bootstrap(a,b,iterations=2000,seed=20261030))


if __name__=='__main__': unittest.main()
