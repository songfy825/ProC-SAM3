"""Label-free manifests and immutable execution identities."""
import json
from pathlib import Path
import random
import re
from .cache import sha256,digest,write_json

FIELDS={'image_id','dataset','source_scene_id','scene_evidence','split','image_path',
        'image_sha256','height','width','city_or_flight'}
LIMITS={'calibration':50,'pilot':50,'confirmation':200}


def read_manifest(path,verify_files=True):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    validate_manifest(rows,verify_files)
    return rows


def validate_manifest(rows,verify_files=True):
    seen_images={};seen_scenes={};seen_ids=set();counts={}
    if not rows: raise ValueError('empty manifest')
    for r in rows:
        if set(r)-FIELDS or FIELDS-{'city_or_flight'}-set(r):
            raise ValueError('manifest schema violation; GT/label fields forbidden')
        if r['dataset'] not in ('LoveDA','iSAID') or r['split'] not in LIMITS:
            raise ValueError('unsupported dataset/split')
        if not r['source_scene_id'] or not r['scene_evidence'].strip():
            raise ValueError('unverified original scene provenance')
        if not re.fullmatch('[0-9a-f]{64}',r['image_sha256']):raise ValueError('invalid image hash')
        if min(r['height'],r['width'])<=0:raise ValueError('invalid image size')
        key=(r['dataset'],r['image_id'])
        if key in seen_ids:raise ValueError('duplicate image id')
        seen_ids.add(key)
        # Byte duplicates across datasets are also prohibited.
        if r['image_sha256'] in seen_images:raise ValueError('image hash overlap')
        seen_images[r['image_sha256']]=r['split']
        scene=(r['dataset'],r['source_scene_id'])
        if scene in seen_scenes and seen_scenes[scene]!=r['split']:
            raise ValueError('original scene leakage across splits')
        seen_scenes[scene]=r['split']
        bucket=(r['dataset'],r['split']);counts[bucket]=counts.get(bucket,0)+1
        if counts[bucket]>LIMITS[r['split']]:raise ValueError('split budget exceeded')
        if verify_files and sha256(r['image_path'])!=r['image_sha256']:
            raise ValueError('input image hash mismatch')
    if len(rows)>600:raise ValueError('total budget exceeded')
    return {'images':len(rows),'independent_source_scenes':len(seen_scenes),
            'cross_split_hash_overlap':0,'cross_split_scene_overlap':0,
            'geographic_independence':'not established by this check'}


def bounded_limits(limits=None):
    limits=LIMITS.copy() if limits is None else limits
    if set(limits)!=set(LIMITS) or any(type(limits[k]) is not int or not 1<=limits[k]<=LIMITS[k] for k in LIMITS):
        raise ValueError('requested split sizes must lie within frozen maxima')
    return limits


def select_splits(inventory,scene_map,seed=20261030,limits=None):
    """External map keys dataset/image_id -> {source_scene_id,scene_evidence}.

    Assign entire scenes before choosing bounded image counts. No GT or metrics.
    """
    limits=bounded_limits(limits);groups={}
    for r in inventory:
        dataset=r['dataset'];identifier=r['image_id'];key=f'{dataset}/{identifier}'
        if key not in scene_map:raise ValueError(f'missing scene provenance: {key}')
        entry=scene_map[key]
        if set(entry)!={'source_scene_id','scene_evidence'} or not all(entry.values()):
            raise ValueError('invalid source scene map')
        groups.setdefault((dataset,entry['source_scene_id']),[]).append(dict(r,**entry))
    rng=random.Random(seed);out=[]
    for dataset in sorted({k[0] for k in groups}):
        keys=sorted(k for k in groups if k[0]==dataset);rng.shuffle(keys)
        if len(keys)<3:raise ValueError('at least three evidenced source scenes needed for splits')
        # Rotate whole scenes through the three pools. Fill each pool up to its cap.
        buckets={s:[] for s in LIMITS}
        for i,key in enumerate(keys):buckets[list(LIMITS)[i%3]].extend(sorted(groups[key],key=lambda r:r['image_id']))
        for split,limit in limits.items():
            out.extend(dict(r,split=split) for r in buckets[split][:limit])
    validate_manifest(out,verify_files=True)
    return out


def create_run(root,run_id,identity):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',run_id):raise ValueError('unsafe run id')
    p=Path(root)/run_id;p.mkdir(parents=True,exist_ok=False)
    write_json(p/'manifest.json',{'schema':1,'run_id':run_id,'identity':identity,
                                'identity_sha256':digest(identity),'status':'IN_PROGRESS'})
    return p


def verify_run(path,identity):
    manifest=json.loads((Path(path)/'manifest.json').read_text())
    if manifest['identity_sha256']!=digest(identity) or manifest['identity']!=identity:
        raise ValueError('run identity changed; create a new run_id')
    return manifest
