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


def preflight(full_images=False):
    source=Path('/home/fengyi/CodeHome/ProC-SAM3-repro');assets={};inventories=[]
    for d in ('LoveDA','iSAID'):
        path=source/f'manifests/{d}-inference.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        if full_images:
            for r in rows:
                if sha256(r['image'])!=r['image_sha256']:raise ValueError(f'image hash mismatch: {r["image"]}')
        inventories.append({'dataset':d,'manifest':str(path),'sha256':sha256(path),'images':len(rows),
                             'all_image_bytes_checked_this_run':full_images,'scene_provenance':'unverified'})
        assets[d]=prompt_inventory(d)
    weight=Path('/home/fengyi/sam3.pt')
    result={'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,
        'cuda_initialized':torch.cuda.is_initialized(),'GPU':'not probed or run by this CPU audit',
        'import_paths':{name:importlib.util.find_spec(name).origin for name in ('torch','sam3','core','proc_research')},
        'source_head':subprocess.check_output(['git','-C',str(source/'src/ProC-SAM3'),'rev-parse','HEAD'],text=True).strip(),
        'source_status':subprocess.check_output(['git','-C',str(source/'src/ProC-SAM3'),'status','--porcelain'],text=True),
        'weight':{'path':str(weight),'bytes':weight.stat().st_size,'sha256':sha256(weight)},
        'datasets':inventories,'published_prompts':assets,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    if result['cuda_initialized']:raise RuntimeError('CPU audit unexpectedly initialized CUDA')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--full-image-hashes',action='store_true');p.add_argument('--output',required=True)
    p=sub.add_parser('inventory');p.add_argument('--output',required=True)
    p=sub.add_parser('capacity');p.add_argument('--manifest',required=True);p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError('audit outputs are immutable')
    if args.command=='preflight':write_json(args.output,preflight(args.full_image_hashes))
    elif args.command=='capacity':
        plan=capacity_plan(read_manifest(args.manifest));write_json(args.output,plan)
        ensure_capacity(args.runtime,plan['estimated_total_bytes'])
    else:
        rows=[]
        for d in ('LoveDA','iSAID'):
            source=Path('/home/fengyi/CodeHome/ProC-SAM3-repro/manifests')/f'{d}-inference.jsonl'
            for line in source.read_text().splitlines():
                r=json.loads(line)
                rows.append({'dataset':d,'image_id':r['id'],'image_path':r['image'],
                    'image_sha256':r['image_sha256'],'height':r['height'],'width':r['width']})
        with open(args.output,'x') as f:
            for r in rows:f.write(json.dumps(r)+'\n')


if __name__=='__main__':main()
