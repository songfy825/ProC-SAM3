"""Portable stage launcher. Never polls for a GPU or starts confirmation implicitly."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .paths import ROOT,interpreter_command
from .cache import sha256,write_json,digest
from .protocol import read_manifest
from .prepare import preflight,capacity_plan
from .cache import ensure_capacity
from .calibration.run import _receipt,identity,_same_experiment


def authorize(stage,allow_gpu,pilot_reviewed):
    if stage=='preflight':return
    if not allow_gpu or os.environ.get('P01_ENABLE_GPU')!='1':
        raise PermissionError('GPU stage requires P01_ENABLE_GPU=1 and --allow-gpu')
    if stage=='confirmation' and not pilot_reviewed:
        raise PermissionError('review pilot before passing --pilot-reviewed for confirmation')


def settings_from(path):
    p=Path(path).resolve();settings=json.loads(p.read_text())
    required={'runtime','checkpoint','manifest','run_prefix'}
    if required-set(settings) or set(settings)-(required|{'pilot_labels','confirmation_labels'}):
        raise ValueError('invalid transfer settings')
    import re
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,70}',settings['run_prefix']):raise ValueError('invalid run prefix')
    for k in set(settings)-{'run_prefix'}:
        value=Path(settings[k]).expanduser()
        settings[k]=str((p.parent/value).resolve() if not value.is_absolute() else value.resolve())
    return settings


def plan(s,stage):
    root=Path(s['runtime']);runs=root/'runs';prefix=s['run_prefix']
    config=root/'inputs'/f'{prefix}-config.json';manifest=s['manifest'];jobs=[]
    def run_path(name):return str(runs/f'{prefix}-{name}')
    def add(name,module,args,output):jobs.append({'stage':name,'module':module,'args':list(map(str,args)),'output':str(output)})
    module='proc_research.calibration.run'
    def gpu(name):
        args=['gpu','--stage',name,'--allow-gpu','--config',config,'--manifest',manifest,
              '--output',runs,'--run-id',f'{prefix}-{name}']
        if name!='g0':args+=['--g0-run',run_path('g0')]
        if name=='confirmation':args+=['--freeze',root/'inputs'/f'{prefix}-seal.json']
        add(name,module,args,run_path(name))
    if stage=='preflight':return []
    if stage in ('g0','pilot'):gpu('g0')
    if stage=='g0':return jobs
    if stage=='pilot':
        gpu('calibration')
        add('fit',module,['fit','--cache-run',run_path('calibration'),'--output',runs,'--run-id',f'{prefix}-banks'],run_path('banks'))
        gpu('pilot')
    elif stage=='confirmation':
        add('freeze',module,['freeze','--config',config,'--manifest',manifest,'--g0-run',run_path('g0'),
            '--pilot-run',run_path('pilot-pred'),'--bank-run',run_path('banks'),'--seal',root/'inputs'/f'{prefix}-seal.json'],root/'inputs'/f'{prefix}-seal.json')
        gpu('confirmation')
    else:raise ValueError('unknown launch stage')
    add('decode',module,['decode','--cache-run',run_path(stage),'--bank-run',run_path('banks'),
        '--output',runs,'--run-id',f'{prefix}-{stage}-pred'],run_path(f'{stage}-pred'))
    labels=s.get(f'{stage}_labels')
    if not labels:raise ValueError(f'{stage}_labels is required for independent evaluation')
    add('evaluate','proc_research.evaluate',['--predictions',Path(run_path(f'{stage}-pred'))/'prediction_index.json',
        '--labels',labels,'--output',run_path(f'{stage}-eval')],run_path(f'{stage}-eval'))
    add('report','proc_research.report',['--prediction-run',run_path(f'{stage}-pred'),
        '--evaluation-dir',run_path(f'{stage}-eval'),'--bank-run',run_path('banks'),'--output',run_path(f'{stage}-report')],run_path(f'{stage}-report'))
    return jobs


def _argument(job,name):
    return Path(job['args'][job['args'].index(name)+1])


def _input_hashes(job):
    """Hash stage metadata only. Actual GT images remain evaluator-only inputs."""
    stage=job['stage'];paths=[]
    if stage in ('fit','decode'):
        paths.append(_argument(job,'--cache-run')/'complete.json')
        if stage=='decode':paths.append(_argument(job,'--bank-run')/'complete.json')
    elif stage=='evaluate':
        paths=[_argument(job,'--predictions'),_argument(job,'--labels')]
    elif stage=='report':
        paths=[_argument(job,'--prediction-run')/'complete.json',
               _argument(job,'--bank-run')/'complete.json',
               _argument(job,'--evaluation-dir')/'provenance.json']
    return {str(p):sha256(p) for p in paths}


def _resume_complete(job,current):
    p=Path(job['output'])
    if not p.exists():return False
    if job['stage']=='freeze':
        seal=json.loads(p.read_text());_same_experiment(current,seal['identity'])
        for path,h in seal['receipts'].items():
            if sha256(path)!=h:raise ValueError('freeze receipt mismatch')
            _receipt(Path(path).parent)
        return True
    if (p/'complete.json').exists():
        manifest,receipt=_receipt(p);ident=manifest['identity']
        _same_experiment(current,ident)
        expected={'g0':'real_gpu_g0','calibration':'real_gpu_cache','pilot':'real_gpu_cache',
                  'confirmation':'real_gpu_cache','fit':'calibration_banks','decode':'predictions'}
        if (ident.get('stage')!=job['stage'] or receipt.get('kind')!=expected.get(job['stage'])
            or ident.get('evidence_kind')!='real_model'):
            raise ValueError('completed run belongs to a different stage or evidence kind')
        for arg,key in (('--cache-run','cache_receipt_sha256'),('--bank-run','bank_receipt_sha256')):
            if arg in job['args']:
                dependency=_argument(job,arg);parent,_=_receipt(dependency)
                _same_experiment(current,parent['identity'])
                if ident.get(key)!=sha256(dependency/'complete.json'):raise ValueError('upstream receipt changed')
                if arg=='--cache-run':
                    split=parent['identity']['stage']
                    if job['stage']=='fit' and split!='calibration':raise ValueError('fit requires calibration')
                    if job['stage']=='decode' and (split not in ('pilot','confirmation') or ident.get('evaluated_split')!=split):
                        raise ValueError('completed decode belongs to another evaluated split')
                elif parent['identity']['stage']!='fit':raise ValueError('invalid bank stage')
        if job['stage']=='confirmation':
            seal_path=_argument(job,'--freeze');seal=json.loads(seal_path.read_text())
            if ident.get('freeze_sha256')!=sha256(seal_path) or ident.get('frozen_bank_receipt_sha256')!=seal['bank_receipt_sha256']:
                raise ValueError('completed confirmation differs from frozen seal/bank')
        return True
    # Evaluations/reports have distinct provenance receipts; never guess complete.
    receipt=p/'launcher_complete.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['experiment_sha256']!=digest(current) or r['job_sha256']!=digest(job):raise ValueError('launcher identity changed')
        if r.get('inputs')!=_input_hashes(job):raise ValueError('evaluation/report input metadata changed')
        for name,h in r['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha256(p/name)!=h:
                raise ValueError('evaluation/report artifact changed')
        if job['stage']=='evaluate':
            # Only this separate evaluator process reads ground-truth image bytes.
            env=dict(os.environ,P01_ENABLE_GPU='0',CUDA_VISIBLE_DEVICES='')
            check=subprocess.run([*interpreter_command(),'-m','proc_research.evaluate',*job['args'],'--verify-only'],
                                 cwd=ROOT,env=env,text=True,capture_output=True)
            if check.returncode:raise ValueError(f'evaluator resume audit failed ({check.returncode}): {check.stderr}')
        return True
    return False


def preserve_incomplete_cpu(job,audit,allowed):
    """Explicit retry keeps failed evidence and does not repeat passed GPU stages."""
    p=Path(job['output'])
    if not allowed:raise FileExistsError(f'incomplete {job["stage"]}: inspect {p}, then explicitly use --retry-cpu')
    if (job['stage'] not in ('fit','decode','evaluate','report') or not p.is_dir()
        or (p/'complete.json').exists() or (p/'launcher_complete.json').exists()):
        raise ValueError('only an incomplete CPU stage may be preserved for retry')
    saved=Path(audit)/('incomplete-'+job['stage'])
    if saved.exists():raise FileExistsError('recovery evidence already exists')
    files={str(f.relative_to(p)):sha256(f) for f in p.rglob('*') if f.is_file()}
    p.rename(saved)
    write_json(Path(audit)/f'recovery-{job["stage"]}.json',{'original':str(p),'preserved':str(saved),'files':files})
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','g0','pilot','confirmation'])
    parser.add_argument('--settings',required=True);parser.add_argument('--allow-gpu',action='store_true')
    parser.add_argument('--pilot-reviewed',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--retry-cpu',action='store_true',help='preserve incomplete CPU output before an explicit retry')
    args=parser.parse_args();s=settings_from(args.settings);jobs=plan(s,args.stage)
    if args.dry_run:print(json.dumps(jobs,indent=2));return
    authorize(args.stage,args.allow_gpu,args.pilot_reviewed)
    os.environ['P01_RUNTIME']=s['runtime'];os.environ['P01_CHECKPOINT']=s['checkpoint']
    root=Path(s['runtime']);(root/'inputs').mkdir(parents=True,exist_ok=True)
    audit=root/'launcher-audits'/f'{s["run_prefix"]}-{time.time_ns()}';audit.mkdir(parents=True)
    # CPU-only preflight even when the requested subsequent stage uses a GPU.
    write_json(audit/'preflight.json',preflight(manifest=s['manifest']))
    rows=read_manifest(s['manifest']);capacity=capacity_plan(rows);write_json(audit/'capacity.json',capacity)
    # Full-budget check before first G0; resumed stages enforce remaining write space.
    first_run=root/'runs'/f'{s["run_prefix"]}-g0'
    ensure_capacity(root,capacity['estimated_total_bytes'] if not first_run.exists() else 0)
    if args.stage=='preflight':print('CPU preflight PASS; GPU has not been tested.');return
    config=root/'inputs'/f'{s["run_prefix"]}-config.json'
    value=json.loads((ROOT/'research/p01/config.json').read_text());value['enabled']=True
    if config.exists():
        if json.loads(config.read_text())!=value:raise ValueError('config changed; choose a new run prefix')
    else:write_json(config,value)
    current=identity(config,s['manifest'],'launcher')
    records=[]
    for i,job in enumerate(jobs):
        if _resume_complete(job,current):print('Verified complete:',job['output'],flush=True);continue
        cmd=[*interpreter_command(),'-m',job['module'],*job['args']]
        if job['stage'] in ('g0','calibration','pilot','confirmation') and (Path(job['output'])/'manifest.json').exists():cmd+=['--resume']
        elif Path(job['output']).exists():
            preserve_incomplete_cpu(job,audit,args.retry_cpu)
        inputs=_input_hashes(job)
        child_env=os.environ.copy()
        if job['stage'] not in ('g0','calibration','pilot','confirmation'):
            child_env.update(P01_ENABLE_GPU='0',CUDA_VISIBLE_DEVICES='')
        log=audit/f'{i:02d}-{job["stage"]}.log'
        with log.open('w') as stream:
            result=subprocess.run(cmd,cwd=ROOT,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
        records.append({'command':cmd,'exit_code':result.returncode,'log':str(log),'log_sha256':sha256(log)})
        write_json(audit/'commands.json',records)
        if result.returncode:raise RuntimeError(f'{job["stage"]} failed ({result.returncode}); see {log}. No automatic retry.')
        if inputs!=_input_hashes(job):raise ValueError('input metadata changed during stage; retain outputs for diagnosis')
        if job['stage'] in ('evaluate','report'):
            p=Path(job['output']);write_json(p/'launcher_complete.json',{'experiment_sha256':digest(current),
                'job_sha256':digest(job),'inputs':inputs,'files':{f.name:sha256(f) for f in p.iterdir() if f.is_file()}})
        print('Completed:',job['stage'],flush=True)


if __name__=='__main__':main()
