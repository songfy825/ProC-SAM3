"""Negative CLI evidence must prove the expected rejection, not just a crash."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from PIL import Image
import numpy as np
from proc_research.cache import sha256
from test_provenance import write_fixture

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get('P02_RUNTIME', str(ROOT/'.p02-runtime'))).expanduser().resolve()


class CLIContracts(unittest.TestCase):
    def test_corrupt_gt_writes_explicit_failure_after_actions(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME/'compat/tmp', prefix='synthetic-cli-') as tmp:
            directory = Path(tmp)
            manifest = write_fixture(directory/'input')
            label = directory/'label.png'
            Image.fromarray(np.ones((4, 8), dtype=np.uint8)).save(label)
            labels = directory/'labels.jsonl'
            labels.write_text(json.dumps({'dataset':'LoveDA','image_id':'synthetic-001',
                'path':str(label),'sha256':'0'*64,
                'image_sha256':sha256(directory/'input/synthetic.png')})+'\n')
            output = directory/'out'
            r = subprocess.run(['bash',str(ROOT/'scripts/p02/python.sh'),
                str(ROOT/'scripts/p02/analyze.py'),'--manifest',str(manifest),
                '--labels',str(labels),'--output',str(output)],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(r.returncode,1,r.stdout+r.stderr)
            self.assertTrue((output/'actions.json').is_file())
            self.assertTrue((output/'status.json').is_file(),r.stdout+r.stderr)
            status=json.loads((output/'status.json').read_text())
            self.assertEqual(status['status'],'FAIL')
            self.assertIn('label content hash mismatch',status['error'])
            self.assertTrue((output/'output_hashes.json').is_file())

    def test_expected_failure_requires_failure_status_and_reason(self):
        import runpy
        verifier = runpy.run_path(str(ROOT/'scripts/p02/verify.py'))
        self.assertIn('check_status',verifier)
        with tempfile.TemporaryDirectory(dir=RUNTIME/'compat/tmp') as tmp:
            directory=Path(tmp)
            check=verifier['check_status']
            self.assertFalse(check(directory,'FAIL','receipt'))
            (directory/'status.json').write_text('{"status":"FAIL","error":"unrelated failure"}')
            self.assertFalse(check(directory,'FAIL','receipt'))
            (directory/'status.json').write_text('{"status":"FAIL","error":"local receipt mismatch"}')
            self.assertTrue(check(directory,'FAIL','receipt'))
            self.assertFalse(check(directory,'PENDING','receipt'))
