"""CPU-only negative audits; each corrupted input must exit 1, restored input 0."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from .cache import SampleCache,write_sample,load_sample,write_json
from .protocol import validate_manifest
from .geometry import validate_encoder


def _fixture():
    s=torch.tensor([.2,.8],dtype=torch.float32)[:,None,None].expand(2,8,12).clone()
    return SampleCache('synthetic','scene','LoveDA',504,s,s.clone(),torch.ones(2),torch.ones(2),
        torch.arange(2),torch.arange(2),(0,1),s.argmax(0),
        {'kind':'full_resize','original_hw':[8,12],'encoder_input':[1,3,504,504],'feature_shapes':[[1,2,36,36]]},
        {k:'a'*64 for k in ('image_sha256','weight_sha256','pool_sha256','config_sha256','code_sha256')})


def check(case,root):
    p=Path(root)
    if case=='cache_corruption':load_sample(p/'sample',json.loads((p/'expected.json').read_text()))
    elif case=='scene_leak':validate_manifest(json.loads((p/'manifest.json').read_text()),False)
    elif case=='encoder_reset':
        x=json.loads((p/'encoder.json').read_text());validate_encoder(x['requested'],x['actual'])
    else:raise ValueError('unknown check')


def negative_checks():
    results={}
    with tempfile.TemporaryDirectory(prefix='p01-negative-') as tmp:
        root=Path(tmp)
        receipt=write_sample(root/'sample',_fixture());write_json(root/'expected.json',receipt)
        rows=[dict(image_id=str(i),dataset='LoveDA',source_scene_id=str(i),scene_evidence='synthetic fixture',
                   split=split,image_path='/not-read',image_sha256=f'{i:064x}',height=8,width=12)
              for i,split in enumerate(('calibration','pilot'),1)]
        write_json(root/'manifest.json',rows)
        write_json(root/'encoder.json',{'requested':504,'actual':[1,3,504,504]})
        for case,path in [('scene_leak',root/'manifest.json'),('cache_corruption',root/'sample/arrays.npz'),('encoder_reset',root/'encoder.json')]:
            original=path.read_bytes()
            if case=='scene_leak':
                bad=json.loads(original);bad[1]['source_scene_id']=bad[0]['source_scene_id'];write_json(path,bad)
            elif case=='cache_corruption':path.write_bytes(original+b'deliberate corruption')
            else:write_json(path,{'requested':504,'actual':[1,3,1008,1008]})
            cmd=[sys.executable,'-m','proc_research.audit','check','--case',case,'--root',str(root)]
            # This read-only interpreter is on a noexec filesystem.
            if sys.platform=='linux':cmd=['/lib64/ld-linux-x86-64.so.2',*cmd]
            negative=subprocess.run(cmd,text=True,capture_output=True)
            path.write_bytes(original)
            restored=subprocess.run(cmd,text=True,capture_output=True)
            if negative.returncode!=1 or restored.returncode!=0:
                raise AssertionError(f'audit not effective: {case}: {negative.returncode}/{restored.returncode}: {negative.stderr} {restored.stderr}')
            results[case]={'command':cmd,'negative_exit':negative.returncode,'negative_stderr':negative.stderr.strip(),
                           'restored_exit':restored.returncode,'restored_stdout':restored.stdout.strip()}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('check');p.add_argument('--case',required=True);p.add_argument('--root',required=True)
    p=sub.add_parser('negative');p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='check':
        try:check(args.case,args.root)
        except ValueError as error:print(str(error),file=sys.stderr);raise SystemExit(1)
        print('PASS: restored input validated')
    else:
        results=negative_checks()
        if Path(args.output).exists():raise FileExistsError('audit results are immutable')
        write_json(args.output,results);print(json.dumps(results,indent=2))


if __name__=='__main__':main()
