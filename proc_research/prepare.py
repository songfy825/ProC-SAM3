"""Read-only asset audit, label-free inventory and conservative capacity planning."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import torch
from .baseline import ROOT,model_options
from .cache import sha256,estimate_cache_bytes,ensure_capacity,write_json
from .protocol import read_manifest
from .paths import checkpoint_path,runtime_root


def prompt_inventory(dataset):
    from core.prompt_bank import PromptBank
    options=model_options(dataset)
    # Pure published-vocabulary expansion only; no constructor/text model call.
    bank=PromptBank.__new__(PromptBank)
    words,ids=bank.get_cls_idx(options['classname_path'])
    with open(options['expanded_prompt_pool_path'],'rb') as f:pool=pickle.load(f)
    queries,mapping=bank._expand_query_words(words,ids,prompt_pool=pool)
    if len(queries)<=len(words):
        raise ValueError('published pool would trigger original exp-file fallback; explicit review needed')
    if set(mapping)!=set(ids):raise ValueError('published pool has missing classes')
    return {'prompts':queries,'prompt_to_class':mapping,'num_prompts':len(queries),'num_classes':max(ids)+1,
            'pool_sha256':sha256(options['expanded_prompt_pool_path']),
            'class_file_sha256':sha256(options['classname_path'])}


def capacity_plan(rows,counts=None):
    counts=counts or {d:(prompt_inventory(d)['num_prompts'],prompt_inventory(d)['num_classes'])
                      for d in sorted({r['dataset'] for r in rows})}
    by_split={}
    for d in counts:
        for split in ('calibration','pilot','confirmation'):
            subset=[r for r in rows if r['dataset']==d and r['split']==split]
            by_split[f'{d}/{split}']=estimate_cache_bytes(subset,*counts[d])
    prediction_bytes=sum(r['height']*r['width']*8*11 for r in rows if r['split']!='calibration')
    cache=sum(by_split.values())
    return {'by_split':by_split,'total_cache_bytes':cache,'prediction_bytes':prediction_bytes,
            'estimated_total_bytes':cache+prediction_bytes,'reserve_bytes':20*1024**3,
            'note':'uncompressed FP32 caches + int64 predictions; signatures/logs extra; no precision fallback'}


def relocated_inventory(rows,dataset,data_root=None):
    out=[]
    for r in rows:
        key=r.get('image_id',r.get('id'))
        if not key or Path(key).is_absolute() or '..' in Path(key).parts:raise ValueError('unsafe image id')
        path=str(Path(data_root)/dataset/'img_dir/val'/key) if data_root else r.get('image_path',r.get('image'))
        out.append(dict(dataset=dataset,image_id=key,image_path=path,image_sha256=r['image_sha256'],height=r['height'],width=r['width']))
    return out


def preflight(full_images=False,manifest=None,manifest_dir=None):
    source=Path(manifest_dir) if manifest_dir else Path('/home/fengyi/CodeHome/ProC-SAM3-repro/manifests')
    assets={};inventories=[]
    selected=read_manifest(manifest,verify_files=True) if manifest else None
    for d in ('LoveDA','iSAID'):
        path=Path(manifest) if manifest else source/f'{d}-inference.jsonl'
        rows=[r for r in selected if r['dataset']==d] if selected else [json.loads(x) for x in path.read_text().splitlines()]
        if full_images and not selected:
            for r in rows:
                if sha256(r['image'])!=r['image_sha256']:raise ValueError(f'image hash mismatch: {r["image"]}')
        inventories.append({'dataset':d,'manifest':str(path),'sha256':sha256(path),'images':len(rows),
                             'all_image_bytes_checked_this_run':bool(selected) or full_images,
                             'scene_provenance':'supplied map; truth requires source review' if selected else 'unverified'})
        assets[d]=prompt_inventory(d)
    weight=checkpoint_path()
    expected=json.loads((ROOT/'research/p01/evidence/assets_initial.json').read_text())['weight']['sha256']
    weight_hash=sha256(weight)
    if weight_hash!=expected:raise ValueError('checkpoint differs from frozen SAM3')
    runtime_root().mkdir(parents=True,exist_ok=True)
    result={'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,
        'cuda_initialized':torch.cuda.is_initialized(),'GPU':'not probed or run by this CPU audit',
        'import_paths':{name:importlib.util.find_spec(name).origin for name in ('torch','sam3','core','proc_research')},
        'checkout_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'checkout_status':subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True),
        'weight':{'path':str(weight),'bytes':weight.stat().st_size,'sha256':weight_hash},
        'datasets':inventories,'published_prompts':assets,'disk_free_bytes':shutil.disk_usage(runtime_root()).free}
    if result['cuda_initialized']:raise RuntimeError('CPU audit unexpectedly initialized CUDA')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--full-image-hashes',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--manifest');p.add_argument('--manifest-dir')
    p=sub.add_parser('inventory');p.add_argument('--output',required=True)
    p.add_argument('--manifest-dir',default='/home/fengyi/CodeHome/ProC-SAM3-repro/manifests');p.add_argument('--data-root')
    p=sub.add_parser('capacity');p.add_argument('--manifest',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError('audit outputs are immutable')
    if args.command=='preflight':write_json(args.output,preflight(args.full_image_hashes,args.manifest,args.manifest_dir))
    elif args.command=='capacity':
        plan=capacity_plan(read_manifest(args.manifest));write_json(args.output,plan)
        ensure_capacity(args.runtime,plan['estimated_total_bytes'])
    else:
        rows=[]
        for d in ('LoveDA','iSAID'):
            source=Path(args.manifest_dir)/f'{d}-inference.jsonl'
            rows.extend(relocated_inventory([json.loads(x) for x in source.read_text().splitlines()],d,args.data_root))
        with open(args.output,'x') as f:
            for r in rows:f.write(json.dumps(r)+'\n')


if __name__=='__main__':main()
