"""Immutable P01 stages. CPU fit/decode; GPU commands require explicit opt-in."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import torch
from ..cache import (sha256,digest,write_json,write_sample,load_sample,
                     estimate_cache_bytes,ensure_capacity)
from ..protocol import read_manifest,create_run,verify_run,select_splits
from ..baseline import ROOT,model_options,require_gpu_authorization
from ..paths import runtime_root
from .signatures import (CalibrationConfig,fit_signatures,shuffle_conditions,
                         save_bank,load_bank)
from .decode import raw_labels,rms_labels,faithful_ecd,decode_multiview

METHODS=('single_raw','single_rms','single_ecd','multiscale_mean','multiscale_rms',
         'multiscale_ecd_global','multiscale_ecd_conditioned','multiscale_ecd_uniform',
         'multiscale_ecd_shuffle0','multiscale_ecd_shuffle1','multiscale_ecd_shuffle2')


def load_config(path):
    c=json.loads(Path(path).read_text())
    if (c['views']!=[1008,504] or c['compile'] or c['decoder_dtype']!='fp32'
        or c['encoder_dtype']!='bf16' or c['shuffle_seeds']!=[0,1,2]
        or tuple(c['methods'])!=METHODS or c['reserve_gib']<20
        or c['split_limits']!={'calibration':50,'pilot':50,'confirmation':200}
        or c['bootstrap_iterations']!=2000 or c['bootstrap_seed']!=20261030):
        raise ValueError('unsupported/unfrozen P01 policy; revise METHOD explicitly')
    CalibrationConfig(**c['calibration'])
    if c['calibration']!=asdict(CalibrationConfig()):
        raise ValueError('calibration constants differ from frozen METHOD v1')
    return c


def code_files():
    # Registration/commit state and evidence updates must not alter code identity.
    paths={'sam3_segmentor_cached.py','research/p01/config.json','requirements.txt'}
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/p01').glob('*') if p.suffix in ('.py','.sh'))
    for folder in ('proc_research','core','sam3','mmengine','configs','prompt_pools'):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py','.txt','.gz','.pkl'))
    return sorted(paths)


def code_identity():
    return digest({f:sha256(ROOT/f) for f in code_files()})


def identity(config_path,manifest_path,stage):
    config=load_config(config_path)
    subprocess.run(['git','diff','--exit-code','234c87e58fb08f02ba1cf163b2a73f5330d462e9','--',
                    'sam3_segmentor_cached.py','core','sam3','configs','prompt_pools'],cwd=ROOT,check=True,capture_output=True)
    assets=json.loads((ROOT/'research/p01/evidence/assets_initial.json').read_text())
    rows=read_manifest(manifest_path,verify_files=False)
    return {'stage':stage,'evidence_kind':'real_model','config':config,'config_sha256':sha256(config_path),
            'base_commit':'234c87e58fb08f02ba1cf163b2a73f5330d462e9',
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'confirmation_images':sorted(r['image_sha256'] for r in rows if r['split']=='confirmation'),
            'manifest_sha256':sha256(manifest_path),'method_sha256':sha256(ROOT/'METHOD.md'),
            'code_sha256':code_identity(),'weight_sha256':assets['weight']['sha256'],
            'pool_sha256':{d:sha256(model_options(d)['expanded_prompt_pool_path']) for d in config['datasets']}}


def _receipt(run):
    p=Path(run);r=json.loads((p/'complete.json').read_text())
    if r.get('status')!='PASS' or 'manifest.json' not in r['files']:
        raise ValueError('run is not complete with a verified manifest')
    for name,h in r['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(p/name)!=h:
            raise ValueError(f'run receipt hash mismatch: {p/name}')
    manifest=json.loads((p/'manifest.json').read_text())
    if manifest['identity_sha256']!=digest(manifest['identity']):raise ValueError('run identity digest mismatch')
    return manifest,r


def finish(run,extra=None):
    p=Path(run)
    if (p/'complete.json').exists():raise FileExistsError('completed receipt is immutable')
    manifest=json.loads((p/'manifest.json').read_text());manifest['status']='PASS'
    write_json(p/'manifest.json',manifest)
    files={str(f.relative_to(p)):sha256(f) for f in p.rglob('*') if f.is_file() and f.name!='complete.json'}
    write_json(p/'complete.json',{'status':'PASS','files':files,**(extra or {})})


def _same_experiment(a,b):
    for k in ('config_sha256','manifest_sha256','method_sha256','code_sha256','weight_sha256','pool_sha256'):
        if a[k]!=b[k]:raise ValueError(f'upstream experiment identity changed: {k}')


def claim_confirmation(root,ident,run_id,seal_hash):
    """One claim per frozen image manifest, independent of seal/config filename.

    Changing parameters cannot make already-exposed confirmation images fresh.
    The production registry is under the fixed experiment P01_RUNTIME root.
    Preserve it when moving machines; changing the root cannot authorize reuse.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    record={'run_id':run_id,'seal_sha256':seal_hash,'config_sha256':ident['config_sha256']}
    keys=ident.get('confirmation_images') or [ident['manifest_sha256']]
    # Lock each image as well as permitting an exact interrupted-run resume.
    # A reordered or partly overlapping manifest cannot recycle held-out images.
    for key in sorted(set(keys)):
        lock=root/(key+'.json')
        if lock.exists() and json.loads(lock.read_text())!=record:
            raise ValueError('confirmation image already claimed; only exact run resume permitted')
    for key in sorted(set(keys)):
        lock=root/(key+'.json')
        try:
            with open(lock,'x') as f:json.dump(record,f,sort_keys=True)
        except FileExistsError:
            if json.loads(lock.read_text())!=record:raise ValueError('concurrent confirmation claim conflict')


def check_stage(stage,ident,g0_run,freeze=None,run_id=None):
    if stage=='g0':return
    if not (Path(g0_run)/'complete.json').is_file():raise ValueError('G0 PASS receipt required before expansion')
    previous,receipt=_receipt(g0_run)
    if previous['identity']['stage']!='g0' or receipt.get('kind')!='real_gpu_g0':raise ValueError('real GPU G0 required')
    _same_experiment(ident,previous['identity'])
    if stage=='confirmation':
        if freeze is None:raise ValueError('sealed pilot/config/bank required for confirmation')
        seal=json.loads(Path(freeze).read_text())
        _same_experiment(ident,seal['identity'])
        for path,h in seal['receipts'].items():
            if sha256(path)!=h:raise ValueError('freeze receipt changed')
            _receipt(Path(path).parent)
        claim_confirmation(runtime_root()/'confirmation_claims',ident,run_id,sha256(freeze))
        ident['frozen_bank_receipt_sha256']=seal['bank_receipt_sha256']
        ident['freeze_sha256']=sha256(freeze)


def validate_bank_binding(ident,bank_receipt_sha256):
    if ident.get('stage')=='confirmation' and ident.get('frozen_bank_receipt_sha256')!=bank_receipt_sha256:
        raise ValueError('confirmation must use the exact frozen calibration bank')


def g0_dataset_complete(run,dataset,rows,have):
    return all((dataset,r['image_id'],v) in have for r in rows for v in (1008,504)) and (Path(run)/f'online_benchmark_{dataset}.json').is_file()


def remaining_cache_bytes(rows,have,prompts,classes):
    return sum(estimate_cache_bytes([r],prompts,classes,
        views=sum((r['dataset'],r['image_id'],v) not in have for v in (1008,504))) for r in rows)


def _cache_entries(run):
    manifest,_=_receipt(run)
    idx=json.loads((Path(run)/'cache_index.json').read_text())
    return manifest,idx['entries']


def _samples(run,entries):
    for entry in entries:
        yield load_sample(Path(run)/entry['path'],expected=entry['receipt'])


def _keys(entries):
    return [SimpleNamespace(key=(e['dataset'],e['scene_id'],e['image_id'],e['view']),view=e['view']) for e in entries]


def gpu_stage(args):
    require_gpu_authorization(args.allow_gpu) # before any CUDA import/check/model construction
    from ..baseline import ProCAdapter
    ident=identity(args.config,args.manifest,args.stage)
    if not ident['config']['enabled']:raise ValueError('config enabled=false; create explicitly enabled frozen config for later GPU run')
    rows=read_manifest(args.manifest)
    selected=[r for r in rows if r['split']==('calibration' if args.stage=='g0' else args.stage)]
    if not selected:raise ValueError('no images in requested stage')
    run=Path(args.output)/args.run_id
    check_stage(args.stage,ident,args.g0_run or '',args.freeze,str(run.resolve()))
    if args.resume:
        verify_run(run,ident)
        if (run/'complete.json').exists():_receipt(run);print('Already complete; hashes verified.');return
        index=json.loads((run/'cache_index.json').read_text())
        for e in index['entries']:load_sample(run/e['path'],expected=e['receipt'])
    else:
        run=create_run(args.output,args.run_id,ident);index={'entries':[],'resources':[]}
        write_json(run/'cache_index.json',index)
    have={(e['dataset'],e['image_id'],e['view']) for e in index['entries']}
    for dataset in ident['config']['datasets']:
        subset=[r for r in selected if r['dataset']==dataset]
        if args.stage=='g0':subset=subset[:2]
        if not subset:raise ValueError(f'empty {dataset} stage')
        cached=all((dataset,r['image_id'],v) in have for r in subset for v in (1008,504))
        if cached and (args.stage!='g0' or g0_dataset_complete(run,dataset,subset,have)):
            continue
        provenance={k:ident[k] for k in ('weight_sha256','config_sha256','code_sha256')}
        provenance['pool_sha256']=ident['pool_sha256'][dataset]
        model=ProCAdapter(dataset,provenance,allow_gpu=True)
        try:
            capacity=ensure_capacity(run,remaining_cache_bytes(subset,have,model.model.num_queries,model.model.num_cls))
            write_json(run/f'capacity_{dataset}.json',capacity)
            for row in subset:
                for view in (1008,504):
                    if (dataset,row['image_id'],view) in have:continue
                    if args.stage=='g0':
                        direct,labels=model.direct(row,view)
                    cache,resource=model.export(row,view)
                    if args.stage=='g0':
                        difference=float((direct-cache.scores).abs().max())
                        if difference>1e-6 or not torch.equal(labels,cache.prediction):
                            raise ValueError(f'G0 direct/export regression: score max error {difference}')
                        repeat,repeat_labels=model.direct(row,view)
                        repeat_error=float((repeat-direct).abs().max())
                        if repeat_error>1e-6 or not torch.equal(labels,repeat_labels):
                            raise ValueError(f'G0 repeat regression: {repeat_error}')
                        resource.update(direct_export_max_abs=difference,repeat_max_abs=repeat_error,labels_equal=True)
                    key=digest([dataset,row['image_id'],view]);path=f'cache/{key}'
                    if (run/path).exists():raise ValueError('orphan cache from interrupted write; retain evidence and use a new run id')
                    start=time.perf_counter();receipt=write_sample(run/path,cache)
                    replay=load_sample(run/path,receipt)
                    if not torch.equal(replay.scores,cache.scores) or not torch.equal(replay.prediction,cache.prediction):
                        raise ValueError('lossless cache roundtrip failed')
                    resource['cache_write_verify_seconds']=time.perf_counter()-start
                    entry=dict(dataset=dataset,image_id=row['image_id'],scene_id=row['source_scene_id'],
                               split=row['split'],view=view,path=path,receipt=receipt)
                    index['entries'].append(entry);index['resources'].append(dict(key=key,**resource))
                    write_json(run/'cache_index.json',index)
            benchmark=run/f'online_benchmark_{dataset}.json'
            if args.stage=='g0' and not benchmark.exists():write_json(benchmark,model.benchmark(subset[0]))
        finally:model.close()
        del model
    write_json(run/'resources.json',{'measurements':index['resources'],'GPU':True,
                                     'cache_speed_is_not_online_speed':True})
    finish(run,{'kind':'real_gpu_g0' if args.stage=='g0' else 'real_gpu_cache'})


def fit_stage(args):
    upstream,entries=_cache_entries(args.cache_run)
    if upstream['identity']['stage']!='calibration' or any(e['split']!='calibration' for e in entries):
        raise ValueError('signature fitting accepts calibration images only')
    ident=dict(upstream['identity'],stage='fit',cache_receipt_sha256=sha256(Path(args.cache_run)/'complete.json'))
    if code_identity()!=ident['code_sha256']:raise ValueError('code changed since inference')
    run=create_run(args.output,args.run_id,ident);index={};start=time.perf_counter()
    cfg=CalibrationConfig(**ident['config']['calibration'])
    for dataset in ident['config']['datasets']:
        items=[e for e in entries if e['dataset']==dataset]
        variants={'single':[e for e in items if e['view']==1008],'multi':items}
        for seed in (0,1,2):variants[f'shuffle{seed}']=items
        index[dataset]={}
        full=None
        for name,subset in variants.items():
            assignment=shuffle_conditions(_keys(subset),int(name[-1])) if name.startswith('shuffle') else None
            bank=fit_signatures(_samples(args.cache_run,subset),cfg,assignment)
            receipt=save_bank(run/dataset/name,bank);index[dataset][name]=receipt
            if name=='multi':full=bank
        # Scene-subset stability diagnostic; no GT or pilot-derived tuning.
        scene_ids=full.meta['scene_ids'];stability=[]
        for fold in (0,1):
            chosen=set(scene_ids[fold::2]);subset=[e for e in items if e['scene_id'] in chosen]
            if not subset:continue
            half=fit_signatures(_samples(args.cache_run,subset),cfg)
            stability.append({'fold':fold,'scenes':len(chosen),
                'Ra_mean_abs_drift':float(np.abs(half.global_bank.R_a-full.global_bank.R_a).mean()),
                'Re_mean_abs_drift':float(np.abs(half.global_bank.R_e-full.global_bank.R_e).mean()),
                'support':{str(v):s.tolist() for v,s in half.support.items()}})
        write_json(run/dataset/'stability.json',stability)
    write_json(run/'bank_index.json',index)
    write_json(run/'resources.json',{'calibration_plus_cache_read_seconds':time.perf_counter()-start,'device':'cpu'})
    finish(run,{'kind':'calibration_banks'})


def decode_stage(args):
    upstream,entries=_cache_entries(args.cache_run);bank_manifest,_=_receipt(args.bank_run)
    _same_experiment(upstream['identity'],bank_manifest['identity'])
    validate_bank_binding(upstream['identity'],sha256(Path(args.bank_run)/'complete.json'))
    if upstream['identity']['stage'] not in ('pilot','confirmation'):raise ValueError('decode stage must be pilot or confirmation')
    if code_identity()!=upstream['identity']['code_sha256']:raise ValueError('code changed since inference')
    ident=dict(upstream['identity'],stage='decode',evaluated_split=upstream['identity']['stage'],
        cache_receipt_sha256=sha256(Path(args.cache_run)/'complete.json'),
        bank_receipt_sha256=sha256(Path(args.bank_run)/'complete.json'))
    run=create_run(args.output,args.run_id,ident)
    bank_index=json.loads((Path(args.bank_run)/'bank_index.json').read_text())
    predictions=[];stats=[];io_time=0;decode_time=0
    for dataset in ident['config']['datasets']:
        banks={name:load_bank(Path(args.bank_run)/dataset/name,receipt) for name,receipt in bank_index[dataset].items()}
        items=[e for e in entries if e['dataset']==dataset];keys=_keys(items)
        shuffled={seed:shuffle_conditions(keys,seed) for seed in (0,1,2)}
        write_json(run/f'shuffle_application_{dataset}.json',{'keys':[list(k.key) for k in keys],'assignments':shuffled})
        grouped={}
        for i,e in enumerate(items):grouped.setdefault(e['image_id'],[]).append((i,e))
        for image_id,pairs in grouped.items():
            pairs.sort(key=lambda p:-p[1]['view'])
            start=time.perf_counter();views=list(_samples(args.cache_run,[e for _,e in pairs]));io_time+=time.perf_counter()-start
            if len(views)!=2:raise ValueError('missing shared view')
            first=views[0];cfg=banks['multi'].config
            start=time.perf_counter()
            results={
                'single_raw':raw_labels(first.scores,cfg.probability_threshold,cfg.background),
                'single_rms':rms_labels([first],[banks['single'].rms[1008]],cfg.probability_threshold,cfg.background),
                'single_ecd':faithful_ecd(first.scores,first.presence,first.active,banks['single'].global_bank,
                                          cfg.sigma,cfg.background,cfg.background_offset),
                'multiscale_mean':raw_labels((views[0].scores+views[1].scores)/2,cfg.probability_threshold,cfg.background),
                'multiscale_rms':rms_labels(views,[banks['multi'].rms[v.view] for v in views],cfg.probability_threshold,cfg.background)}
            results['single_ecd'][first.scores.amax(0)<cfg.probability_threshold]=cfg.background
            for name,conditional,prior in [('global',False,'quality'),('conditioned',True,'quality'),('uniform',True,'uniform')]:
                method=f'multiscale_ecd_{name}'
                results[method],s=decode_multiview(views,banks['multi'],conditional=conditional,prior_mode=prior)
                stats.append(dict(dataset=dataset,image_id=image_id,method=method,**s))
            for seed in (0,1,2):
                method=f'multiscale_ecd_shuffle{seed}'
                conditions=[shuffled[seed][i] for i,_ in pairs]
                results[method],s=decode_multiview(views,banks[f'shuffle{seed}'],True,conditions)
                stats.append(dict(dataset=dataset,image_id=image_id,method=method,**s))
            decode_time+=time.perf_counter()-start
            for method,labels in results.items():
                path=run/'predictions'/dataset/method/(digest(image_id)+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,labels.numpy(),allow_pickle=False)
                predictions.append(dict(dataset=dataset,image_id=image_id,scene_id=first.scene_id,
                    method=method,split=ident['evaluated_split'],num_classes=len(first.class_ids),
                    path=str(path.resolve()),sha256=sha256(path)))
    write_json(run/'prediction_index.json',{'identity':ident,'predictions':predictions})
    write_json(run/'fallback.json',stats)
    write_json(run/'resources.json',{'cache_read_seconds':io_time,'cpu_decode_seconds':decode_time,
                                    'online_inference_seconds':None,'note':'cache replay does not measure online latency'})
    finish(run,{'kind':'predictions'})


def freeze_stage(args):
    ident=identity(args.config,args.manifest,'freeze')
    receipts={}
    for path,kind in ((args.g0_run,'real_gpu_g0'),(args.pilot_run,'predictions'),(args.bank_run,'calibration_banks')):
        manifest,receipt=_receipt(path);_same_experiment(ident,manifest['identity'])
        if receipt['kind']!=kind:raise ValueError('freeze stage receipt mismatch')
        if kind=='predictions' and manifest['identity'].get('evaluated_split')!='pilot':raise ValueError('pilot required')
        receipts[str((Path(path)/'complete.json').resolve())]=sha256(Path(path)/'complete.json')
        if kind=='predictions' and manifest['identity']['bank_receipt_sha256']!=sha256(Path(args.bank_run)/'complete.json'):
            raise ValueError('freeze bank differs from the evaluated pilot bank')
    # Freeze can follow negative pilot: average need not improve for ECD to continue.
    p=Path(args.seal);p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x') as f:json.dump({'identity':ident,'receipts':receipts,
        'bank_receipt_sha256':sha256(Path(args.bank_run)/'complete.json'),'scientific_status':'PENDING'},f,indent=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('gpu','fit','decode'):
        p=sub.add_parser(command);p.add_argument('--output',required=True);p.add_argument('--run-id',required=True)
        if command=='gpu':
            p.add_argument('--stage',choices=['g0','calibration','pilot','confirmation'],required=True)
            p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
            p.add_argument('--g0-run');p.add_argument('--freeze');p.add_argument('--allow-gpu',action='store_true')
            p.add_argument('--resume',action='store_true')
        else:
            p.add_argument('--cache-run',required=True)
            if command=='decode':p.add_argument('--bank-run',required=True)
    p=sub.add_parser('freeze')
    for name in ('config','manifest','g0-run','pilot-run','bank-run','seal'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('split');p.add_argument('--inventory',required=True);p.add_argument('--scene-map',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limits',help='JSON object with smaller calibration/pilot/confirmation image counts per dataset')
    args=parser.parse_args()
    if args.command=='gpu':gpu_stage(args)
    elif args.command=='fit':fit_stage(args)
    elif args.command=='decode':decode_stage(args)
    elif args.command=='freeze':freeze_stage(args)
    else:
        rows=[json.loads(x) for x in Path(args.inventory).read_text().splitlines() if x.strip()]
        chosen=select_splits(rows,json.loads(Path(args.scene_map).read_text()),limits=json.loads(args.limits) if args.limits else None)
        with open(args.output,'x') as f:
            for row in chosen:f.write(json.dumps(row)+'\n')


if __name__=='__main__':main()
