import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


class Transfer(unittest.TestCase):
    def test_launcher_runtime_follows_moved_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            moved=Path(d)/'moved project';script=moved/'scripts/p01/python.sh'
            script.parent.mkdir(parents=True);shutil.copyfile(ROOT/'scripts/p01/python.sh',script)
            env=os.environ.copy();env.pop('P01_RUNTIME',None);env.pop('CONDA_PREFIX',None)
            env['P01_PYTHON']=sys.executable
            r=subprocess.run(['bash',str(script),'-c','import os,json; print(json.dumps([os.environ["P01_RUNTIME"],os.environ["CUDA_VISIBLE_DEVICES"]]))'],env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(r.stdout),[str(moved/'.p01-runtime'),''])
    def test_checkpoint_and_runtime_paths_are_explicit(self):
        from proc_research.paths import checkpoint_path,runtime_root
        with patch.dict(os.environ,{'P01_CHECKPOINT':'/transfer/sam3.pt','P01_RUNTIME':'/transfer/run'}):
            self.assertEqual(str(checkpoint_path()),'/transfer/sam3.pt')
            self.assertEqual(str(runtime_root()),'/transfer/run')
    def test_inventory_relocation_preserves_ids_and_hash(self):
        from proc_research.prepare import relocated_inventory
        rows=[{'id':'P1.png','image':'/old/iSAID/img_dir/val/P1.png','image_sha256':'a'*64,'height':8,'width':12}]
        result=relocated_inventory(rows,'iSAID',Path('/new/data'))
        self.assertEqual(result[0]['image_path'],'/new/data/iSAID/img_dir/val/P1.png')
        self.assertEqual(result[0]['image_sha256'],'a'*64)
    def test_launch_plan_has_g0_before_expansion_and_separate_confirmation(self):
        from proc_research.launch import plan
        settings={'runtime':'/tmp/p01-transfer-test','checkpoint':'/weights/sam3.pt','manifest':'/input/splits.jsonl',
                  'run_prefix':'test4090','pilot_labels':'/eval/pilot.jsonl','confirmation_labels':'/eval/confirmation.jsonl'}
        jobs=plan(settings,'pilot')
        self.assertEqual(jobs[0]['stage'],'g0')
        self.assertNotIn('confirmation',[j['stage'] for j in jobs])
        self.assertIn('evaluate',[j['stage'] for j in jobs])
        self.assertEqual(plan(settings,'confirmation')[0]['stage'],'freeze')
    def test_launch_requires_explicit_gpu_authorization(self):
        from proc_research.launch import authorize
        with self.assertRaises(PermissionError):authorize('pilot',False,False)
        with self.assertRaises(PermissionError):authorize('confirmation',True,False)
        authorize('preflight',False,False)
    def test_package_repair_uses_archive_link_metadata(self):
        from scripts.p01.pack_environment import restore_member
        import tarfile
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);(root/'lib/python3.12').mkdir(parents=True);(root/'lib/python3.1').mkdir()
            member=tarfile.TarInfo('lib/python3.1');member.type=tarfile.SYMTYPE;member.linkname='python3.12';member.mode=0o777
            restore_member(root,member)
            self.assertTrue((root/'lib/python3.1').is_symlink())
            self.assertEqual((root/'lib/python3.1').resolve(),root/'lib/python3.12')
    def test_resume_evaluation_rejects_changed_label_manifest(self):
        from proc_research.cache import write_json,sha256,digest
        from proc_research.evaluate import evaluate_predictions
        from proc_research.launch import _resume_complete
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'eval';target=root/'gt.png';prediction=root/'pred.npy'
            Image.fromarray(np.zeros((2,3),dtype='uint8')).save(target)
            np.save(prediction,np.zeros((2,3),dtype='int64'))
            labels=root/'labels.jsonl';labels.write_text(json.dumps(dict(dataset='iSAID',image_id='i',path=str(target),sha256=sha256(target)))+'\n')
            original_labels=labels.read_bytes()
            predictions=root/'prediction_index.json'
            write_json(predictions,{'predictions':[dict(dataset='iSAID',image_id='i',scene_id='s',split='pilot',
                method='single_raw',num_classes=2,path=str(prediction),sha256=sha256(prediction))]})
            evaluate_predictions(predictions,labels,output)
            job={'stage':'evaluate','module':'proc_research.evaluate','args':['--predictions',str(predictions),
                 '--labels',str(labels),'--output',str(output)],'output':str(output)}
            current={'synthetic_fixture':True}
            write_json(output/'launcher_complete.json',{'experiment_sha256':digest(current),'job_sha256':digest(job),
                'inputs':{str(labels):sha256(labels),str(predictions):sha256(predictions)},
                'files':{f.name:sha256(f) for f in output.iterdir()}})
            self.assertTrue(_resume_complete(job,current))
            labels.write_text('changed label metadata\n')
            with self.assertRaises(ValueError):_resume_complete(job,current)
            labels.write_bytes(original_labels)
            target.write_bytes(target.read_bytes()+b'corruption')
            with self.assertRaises(ValueError):_resume_complete(job,current)
    def test_resume_rejects_wrong_stage_receipt(self):
        from proc_research.calibration.run import finish
        from proc_research.protocol import create_run
        from proc_research.launch import _resume_complete
        with tempfile.TemporaryDirectory() as d:
            ident={k:'a'*64 for k in ('config_sha256','manifest_sha256','method_sha256','code_sha256','weight_sha256','pool_sha256')}
            run=create_run(d,'wrong',dict(ident,stage='fit',evidence_kind='synthetic_cpu'))
            finish(run,{'kind':'calibration_banks'})
            with self.assertRaises(ValueError):
                _resume_complete({'stage':'g0','output':str(run),'args':[]},ident)
    def test_resume_decode_is_bound_to_split_and_upstream(self):
        from proc_research.calibration.run import finish
        from proc_research.protocol import create_run
        from proc_research.cache import sha256
        from proc_research.launch import _resume_complete
        with tempfile.TemporaryDirectory() as d:
            ident={k:'a'*64 for k in ('config_sha256','manifest_sha256','method_sha256','code_sha256','weight_sha256','pool_sha256')}
            # Receipt validation fixture only; no model or metric output is created.
            ident['evidence_kind']='real_model'
            cache=create_run(d,'pilot',dict(ident,stage='pilot'));finish(cache,{'kind':'real_gpu_cache'})
            bank=create_run(d,'banks',dict(ident,stage='fit'));finish(bank,{'kind':'calibration_banks'})
            for split in ('pilot','confirmation'):
                pred=create_run(d,split+'-pred',dict(ident,stage='decode',evaluated_split=split,
                    cache_receipt_sha256=sha256(cache/'complete.json'),bank_receipt_sha256=sha256(bank/'complete.json')))
                finish(pred,{'kind':'predictions'})
                job={'stage':'decode','output':str(pred),'args':['--cache-run',str(cache),'--bank-run',str(bank)]}
                if split=='pilot':self.assertTrue(_resume_complete(job,ident))
                else:
                    with self.assertRaises(ValueError):_resume_complete(job,ident)
    def test_cpu_retry_preserves_incomplete_evidence(self):
        from proc_research.launch import preserve_incomplete_cpu
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'eval';output.mkdir();audit=root/'audit';audit.mkdir()
            (output/'partial.json').write_text('retained partial result')
            job={'stage':'evaluate','output':str(output)}
            with self.assertRaises(FileExistsError):preserve_incomplete_cpu(job,audit,False)
            saved=preserve_incomplete_cpu(job,audit,True)
            self.assertFalse(output.exists())
            self.assertEqual((saved/'partial.json').read_text(),'retained partial result')
            output.mkdir();(output/'complete.json').write_text('{}')
            with self.assertRaises(ValueError):preserve_incomplete_cpu(job,audit,True)


if __name__=='__main__':unittest.main()
