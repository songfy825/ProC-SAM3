"""Run the complete CPU acceptance suite and persist exact commands/exit codes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['P01_ENABLE_GPU']='0'
    from proc_research.calibration.run import code_identity
    from proc_research.cache import sha256
    python=['/lib64/ld-linux-x86-64.so.2',sys.executable]
    commands=[
        ('contracts',python+['-m','unittest','discover','-s','tests/research','-v']),
        ('negative_audits',python+['-m','proc_research.audit','negative','--output',str(output/'negative.json')]),
        ('frozen_contract',['sha256sum','-c','research/p01/evidence/contract-freeze.sha256']),
        ('whitespace',['git','diff','--check']),
        ('original_model_unchanged',['git','diff','--exit-code','234c87e58fb08f02ba1cf163b2a73f5330d462e9',
                                    '--','sam3_segmentor_cached.py','core','sam3','configs','prompt_pools']),
    ]
    for module in ('proc_research.calibration.run','proc_research.evaluate','proc_research.report','proc_research.prepare','proc_research.audit'):
        commands.append((module.replace('.','_')+'_help',python+['-m',module,'--help']))
    records=[]
    for name,cmd in commands:
        start=time.perf_counter();r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
        log=output/(name+'.log');log.write_text(r.stdout+r.stderr)
        records.append({'name':name,'command':cmd,'cwd':str(root),'exit_code':r.returncode,
            'seconds':time.perf_counter()-start,'log':str(log),'log_sha256':sha256(log)})
        print(name, 'exit',r.returncode,flush=True)
    text=(output/'contracts.log').read_text();match=re.search(r'Ran (\d+) tests',text)
    evidence={'scope':'CPU synthetic contracts only; GPU/real research PENDING',
        'time_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':code_identity(),
        'method_sha256':sha256(root/'METHOD.md'),'config_sha256':sha256(root/'research/p01/config.json'),
        'tests':int(match.group(1)) if match else None,'commands':records,
        'test_source_sha256':{str(p.relative_to(root)):sha256(p) for p in sorted((root/'tests/research').glob('test_*.py'))},
        'status':'PASS' if match and all(r['exit_code']==0 for r in records) else 'FAIL'}
    (output/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if evidence['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
