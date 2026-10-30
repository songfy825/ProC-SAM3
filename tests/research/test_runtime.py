import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
import torch


class RuntimeContracts(unittest.TestCase):
    def test_gpu_disabled_without_both_explicit_switches(self):
        from proc_research.baseline import require_gpu_authorization
        with self.assertRaises(PermissionError):require_gpu_authorization(False)
        if os.environ.get('P01_ENABLE_GPU')!='1':
            with self.assertRaises(PermissionError):require_gpu_authorization(True)
        self.assertFalse(torch.cuda.is_initialized())
    def test_default_config_freezes_required_controls(self):
        from proc_research.calibration.run import load_config
        c=load_config(Path(__file__).resolve().parents[2]/'research/p01/config.json')
        self.assertEqual(c['views'],[1008,504]);self.assertFalse(c['enabled'])
        self.assertFalse(c['compile']);self.assertEqual(c['shuffle_seeds'],[0,1,2])
        self.assertIn('multiscale_ecd_uniform',c['methods'])
    def test_missing_pool_rejected_before_model_import(self):
        from proc_research.baseline import model_options
        with self.assertRaises(FileNotFoundError):model_options('LoveDA',pool='/absent/pool.pkl')
    def test_changed_published_pool_rejected(self):
        from proc_research.baseline import model_options
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'changed.pkl';p.write_bytes(b'not the frozen released pool')
            with self.assertRaises(ValueError):model_options('LoveDA',pool=p)
    def test_adapter_retains_algorithm_methods(self):
        from proc_research.baseline import author_source
        root=Path(__file__).resolve().parents[2]
        raw=(root/'sam3_segmentor_cached.py').read_text()
        adapted=author_source('/home/fengyi/sam3.pt',str(root/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
        def methods(src):
            cls=next(x for x in ast.parse(src).body if isinstance(x,ast.ClassDef))
            return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name!='__init__'}
        self.assertEqual(methods(raw),methods(adapted))
        self.assertNotIn('from mmseg',adapted)
    def test_original_reducer_order_logit_then_sigmoid(self):
        from core.result_reducer import GroundingResultReducer
        reducer=GroundingResultReducer('cpu',use_transformer_decoder=False,use_presence_score=False)
        outputs={'semantic_seg':torch.tensor([[[[-4.,4.],[-4.,4.]]]]),'presence_logit_dec':torch.zeros(1)}
        result=reducer.reduce_slot_outputs(outputs,1,(2,3),True)
        torch.testing.assert_close(result['semantic_logits'][0,:,1],torch.full((2,),.5))
        torch.testing.assert_close(result['presence_scores'],torch.tensor([.5]))
    def test_confirmation_requires_sealed_config_and_receipts(self):
        from proc_research.calibration.run import check_stage
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):check_stage('confirmation',{},Path(d))
    def test_negative_audit_entrypoint(self):
        from proc_research.audit import negative_checks
        result=negative_checks()
        self.assertEqual(set(result),{'scene_leak','cache_corruption','encoder_reset'})
        for value in result.values():
            self.assertEqual(value['negative_exit'],1);self.assertEqual(value['restored_exit'],0)


if __name__=='__main__':unittest.main()
