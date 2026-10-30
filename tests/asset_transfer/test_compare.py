import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Compare(unittest.TestCase):
    def test_omitted_dataset_root_cannot_pass(self):
        script=Path(__file__).resolve().parents[2]/'scripts/asset_transfer/compare_assets.py'
        expected=[dict(dataset='LoveDA',relative_path='img_dir/val/a.png',bytes=1,sha256='a'*64)]
        result=subprocess.run([sys.executable,str(script),'--roots','{}'],input=json.dumps(expected),text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_readonly_comparison_rejects_missing_corrupt_and_extra_files(self):
        script=Path(__file__).resolve().parents[2]/'scripts/asset_transfer/compare_assets.py'
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);p=root/'img_dir/val/a.png';p.parent.mkdir(parents=True);p.write_bytes(b'original')
            expected=[dict(dataset='LoveDA',relative_path='img_dir/val/a.png',bytes=8,sha256=hashlib.sha256(b'original').hexdigest())]
            def run():
                r=subprocess.run([sys.executable,str(script),'--roots',json.dumps({'LoveDA':str(root)})],
                                 input=json.dumps(expected),text=True,capture_output=True)
                return r.returncode,json.loads(r.stdout)
            code,result=run();self.assertEqual(code,0)
            self.assertEqual(result['actual_canonical_sha256'],result['expected_canonical_sha256'])
            p.write_bytes(b'corrupte');code,result=run();self.assertEqual(code,1)
            self.assertEqual(result['datasets'][0]['hash_mismatch'],['img_dir/val/a.png'])
            p.write_bytes(b'original');extra=p.with_name('extra.png');extra.write_bytes(b'extra')
            code,result=run();self.assertEqual(code,1);self.assertEqual(result['datasets'][0]['extra_files'],['img_dir/val/extra.png'])
            extra.unlink();p.unlink();code,result=run();self.assertEqual(code,1)
            self.assertEqual(result['datasets'][0]['missing'],['img_dir/val/a.png'])
            self.assertEqual(list(p.parent.iterdir()),[])


if __name__=='__main__':unittest.main()
