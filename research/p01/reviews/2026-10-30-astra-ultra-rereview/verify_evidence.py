"""Small-file provenance checks only. No torch, model, dataset or weight access."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/fengyi/CodeHome/Proc/src/ProC-SAM3')
OUT=Path(__file__).resolve().parent
E=ROOT/'research/p01/p2-fixes/evidence'
checks=[]
def sha(p):
    p=Path(p)
    assert p.stat().st_size < 20*1024**2, f'unexpected large evidence: {p}'
    return hashlib.sha256(p.read_bytes()).hexdigest()
def check(p,expected):
    actual=sha(p)
    checks.append({'path':str(p),'sha256':actual,'matches':actual==expected})
    assert actual==expected, p

head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
assert head=='4013778dc6bd476bfa5182514498747e77c08ffe'
assert not status
check(ROOT/'research/p01/reviews/2026-10-30-astra-ultra/REVIEW.md','bbf4370c1e900595f0f7a0b3e44126ccaf0417369278968feb0f1f4aa688c513')
freeze=json.loads((E/'test-freeze.json').read_text())
check(ROOT/freeze['file'],freeze['sha256'])
assert freeze['sha256']=='33a36d9a6365d83114d81560ab6168cd57e1b200554af420749b45dabb5884f4'
for name in ('red','green'):
    r=json.loads((E/(name+'.json')).read_text())
    assert r['test_sha256']==freeze['sha256']
    assert r['exit_code']==(1 if name=='red' else 0)
for name in ('test-freeze','red','f1-green','green'):
    check(Path('/home/fengyi/CodeHome/Proc/runtime/p01/audit/p2-fixes-v1')/(name+'.json'),sha(E/(name+'.json')))
catalog=json.loads((E/'log-catalog.json').read_text())
for p,r in catalog.items():
    check(p,r['sha256'])
    assert Path(p).stat().st_size==r['bytes']
red=Path('/home/fengyi/CodeHome/Proc/runtime/p01/audit/p2-fixes-v1/red.log').read_text()
assert 'Ran 10 tests' in red and 'FAILED (failures=9)' in red and 'ERROR:' not in red
green=Path('/home/fengyi/CodeHome/Proc/runtime/p01/audit/p2-fixes-v1/green.log').read_text()
assert 'Ran 10 tests' in green and '\nOK\n' in green
v=json.loads((E/'verification.json').read_text())
check(E/'verification.json','0e0b19a7b107e0e969d8145690e76d490dc857ca3a4af030cd66056239ae7098')
check(Path('/home/fengyi/CodeHome/Proc/runtime/p01/audit/cpu-p2-v1/verification.json'),sha(E/'verification.json'))
assert v['status']=='PASS' and v['tests']==70
for r in v['commands']:
    assert r['exit_code']==0
    check(r['log'],r['log_sha256'])
for p,h in v['test_source_sha256'].items():check(ROOT/p,h)
contracts=Path(next(r['log'] for r in v['commands'] if r['name']=='contracts')).read_text()
assert 'Ran 70 tests' in contracts and '\nOK\n' in contracts and 'skipped=' not in contracts
for r in json.loads((E/'negative.json').read_text()).values():
    assert r['negative_exit']==1 and r['restored_exit']==0
for key,path in [('method_sha256','METHOD.md'),('config_sha256','research/p01/config.json')]:check(ROOT/path,v[key])
paths={'sam3_segmentor_cached.py','research/p01/config.json','requirements.txt'}
paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/p01').glob('*') if p.suffix in ('.py','.sh'))
for folder in ('proc_research','core','sam3','mmengine','configs','prompt_pools'):
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.txt','.gz','.pkl'))
production_hash=hashlib.sha256(json.dumps({p:sha(ROOT/p) for p in sorted(paths)},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert production_hash==v['code_sha256']=='df52a7636ed6fb0de5a613ceea513c68632fdcca7f61c8d415ce900d45caab59'
result={'status':'PASS','head':head,'git_status':status,'historical_CPU_suite':{'tests':70,'all_commands_exit_zero':True,'no_skips':True},
        'RED':{'tests':10,'failures':9,'errors':0,'exit_code':1},'GREEN':{'tests':10,'exit_code':0},
        'code_sha256':production_hash,'code_files_hashed':len(paths),'evidence_checks':checks,
        'scope':'Small evidence and production source only. No datasets, weights, torch import or GPU activity.'}
(OUT/'evidence-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='evidence_checks'},indent=2))
print('evidence_hash_matches',len(checks))
